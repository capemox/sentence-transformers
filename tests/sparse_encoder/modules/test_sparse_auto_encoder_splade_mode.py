from __future__ import annotations

import tempfile
from contextlib import nullcontext

import pytest
import torch

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.modules import (
    SparseAutoEncoder,
    SpladePooling,
    Transformer,
)

BACKBONE = "sentence-transformers-testing/stsb-bert-tiny-safetensors"


@pytest.fixture(scope="module")
def sae_splade_model() -> SparseEncoder:
    """Build the canonical SAE-SPLADE pipeline on a tiny BERT.

    ``Transformer(feature-extraction)`` -> :class:`SparseAutoEncoder` (mode=splade) -> :class:`SpladePooling`.
    """
    transformer = Transformer(BACKBONE, transformer_task="feature-extraction")
    sae = SparseAutoEncoder(
        input_dim=transformer.get_embedding_dimension(),
        hidden_dim=64,
        k=4,
        k_aux=2,
        dead_threshold=2,
        mode="splade",
    )
    pool = SpladePooling(pooling_strategy="max")
    return SparseEncoder(modules=[transformer, sae, pool])


def _to_dense(t: torch.Tensor) -> torch.Tensor:
    return t.to_dense() if t.is_sparse else t


def test_pipeline_shapes_and_sparsity(sae_splade_model: SparseEncoder) -> None:
    emb = _to_dense(sae_splade_model.encode(["hello world", "this is a longer test sentence"]))
    # Aggregated vector lives in the SAE's hidden_dim space, not the backbone's vocab.
    assert tuple(emb.shape) == (2, sae_splade_model[1].hidden_dim)
    assert (emb >= 0).all(), "ReLU + log1p means every entry is non-negative"
    # The aggregated vector is still meaningfully sparse — far below `hidden_dim`.
    assert int((emb > 0).sum()) < emb.numel()


def test_mode_argument_validation() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        SparseAutoEncoder(input_dim=8, mode="not-a-mode")


@pytest.mark.parametrize(
    ["is_inference", "expected_extra_keys"],
    [
        (True, set()),
        (
            False,
            {
                "token_embeddings_backbone",
                "token_embeddings_encoded",
                "token_embeddings_encoded_4k",
                "auxiliary_token_embeddings",
                "decoded_token_embeddings_k",
                "decoded_token_embeddings_4k",
                "decoded_token_embeddings_aux",
                "decoded_token_embeddings_k_pre_bias",
            },
        ),
    ],
)
def test_training_intermediates_use_token_prefix(
    sae_splade_model: SparseEncoder, is_inference: bool, expected_extra_keys: set
) -> None:
    """In splade mode the training-time intermediates must be exposed under a
    ``token_embeddings_*`` / ``decoded_token_embeddings_*`` prefix so they're
    distinguishable from the csr-mode ``sentence_embedding_*`` outputs that downstream
    losses are written against."""
    inputs = sae_splade_model.preprocess(["intermediate exposure test"])
    inputs = {k: v.to(sae_splade_model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.inference_mode() if is_inference else nullcontext():
        out = sae_splade_model(inputs)

    base_keys = {"input_ids", "attention_mask", "token_type_ids", "token_embeddings", "sentence_embedding", "modality"}
    assert base_keys.issubset(out.keys())
    assert set(out.keys()) - base_keys == expected_extra_keys

    if not is_inference:
        seq_len = inputs["input_ids"].shape[1]
        hidden = sae_splade_model[0].get_embedding_dimension()
        # Decoded tensors round-trip back to the backbone hidden size at the token level.
        assert tuple(out["decoded_token_embeddings_k"].shape) == (1, seq_len, hidden)
        assert out["auxiliary_token_embeddings"].shape[-1] == sae_splade_model[1].hidden_dim


def test_k_zero_disables_topk_mask() -> None:
    """``k=0`` is the SPLADE fine-tuning path: no top-K mask, sparsity comes from
    ReLU + log1p + max-pool only. Verify it (a) runs and (b) skips the AuxK output."""
    transformer = Transformer(BACKBONE, transformer_task="feature-extraction")
    sae = SparseAutoEncoder(
        input_dim=transformer.get_embedding_dimension(),
        hidden_dim=32,
        k=0,
        k_aux=4,
        mode="splade",
    )
    model = SparseEncoder(modules=[transformer, sae, SpladePooling(pooling_strategy="max")])

    inputs = model.preprocess(["k=0 path"])
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    out = model(inputs)

    # token-level SAE output spans the full dictionary, not just k positions
    assert out["token_embeddings"].shape[-1] == 32
    # 4k path collapses to k path when k=0 (no slicing happens)
    assert torch.equal(out["token_embeddings_encoded_4k"], out["token_embeddings"])
    # AuxK is meaningless without a top-K mask; skipped
    assert out["auxiliary_token_embeddings"] is None


def test_max_active_dims_forward_kwarg(sae_splade_model: SparseEncoder) -> None:
    """``max_active_dims`` overrides ``k`` at call time (used to tighten queries vs docs).

    The per-token top-K count must equal the override, regardless of what happens after
    max-pool — that's the contract callers rely on for query/doc top-K asymmetry.
    """
    inputs = sae_splade_model.preprocess(["query top-k override test"])
    inputs = {k: v.to(sae_splade_model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.inference_mode():
        default = sae_splade_model(inputs)
        tighter = sae_splade_model(dict(inputs), max_active_dims=2)

    per_token_nnz_default = (default["token_embeddings"] > 0).sum(dim=-1)
    per_token_nnz_tighter = (tighter["token_embeddings"] > 0).sum(dim=-1)
    assert torch.all(per_token_nnz_default <= sae_splade_model[1].k)
    assert torch.all(per_token_nnz_tighter <= 2)
    # A tighter top-K can never enlarge the pooled vector's support.
    assert int((tighter["sentence_embedding"] > 0).sum()) <= int((default["sentence_embedding"] > 0).sum())


def test_save_and_reload(sae_splade_model: SparseEncoder, tmp_path) -> None:
    inputs = ["save reload check"]
    before = _to_dense(sae_splade_model.encode(inputs))

    with tempfile.TemporaryDirectory(dir=tmp_path) as out:
        sae_splade_model.save_pretrained(out)
        reloaded = SparseEncoder(out)

    # ``mode`` is a config_key, so it must round-trip through modules.json.
    reloaded_sae = next(m for m in reloaded if isinstance(m, SparseAutoEncoder))
    assert reloaded_sae.mode == "splade"
    after = _to_dense(reloaded.encode(inputs))
    torch.testing.assert_close(before, after)


def test_dead_neuron_stats_update(sae_splade_model: SparseEncoder) -> None:
    """``stats_last_nonzero`` is the buffer the AuxK mask consults to find dead latents.
    Each training-mode forward must update it; the values must keep moving so dead
    latents can rotate in and out of the AuxK pool."""
    sae = sae_splade_model[1]
    assert isinstance(sae, SparseAutoEncoder)
    sae.stats_last_nonzero.zero_()

    inputs = sae_splade_model.preprocess(["dead neuron stats"])
    inputs = {k: v.to(sae_splade_model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    sae_splade_model(inputs)
    after_one = sae.stats_last_nonzero.clone()
    assert (after_one > 0).all(), "every latent's step counter must advance on each forward"

    sae_splade_model(inputs)
    assert not torch.equal(sae.stats_last_nonzero, after_one), "stats buffer must keep moving"


# -- splade-mode-specific architecture: untied decoder, row-normalisation, corpus stats --


def test_splade_mode_has_untied_decoder() -> None:
    """splade mode allocates a separate ``W_dec`` parameter and disables the tied
    decoder; csr mode keeps the existing tied decoder and has no ``W_dec``."""
    splade = SparseAutoEncoder(input_dim=8, hidden_dim=16, k=4, k_aux=4, mode="splade")
    assert isinstance(splade.W_dec, torch.nn.Parameter)
    assert tuple(splade.W_dec.shape) == (16, 8)
    assert splade.decoder is None  # tied decoder explicitly removed

    csr = SparseAutoEncoder(input_dim=8, hidden_dim=16, k=4, mode="csr")
    assert not hasattr(csr, "W_dec") or getattr(csr, "W_dec", None) is None
    # The tied decoder is still in place for csr mode.
    assert csr.decoder is not None


def test_splade_decoder_rows_unit_norm_at_init() -> None:
    """Each ``W_dec`` row is L2-normalised at construction so concepts start on
    the unit sphere — matches the paper's init."""
    sae = SparseAutoEncoder(input_dim=8, hidden_dim=16, k=4, k_aux=4, mode="splade")
    torch.testing.assert_close(sae.W_dec.norm(dim=-1), torch.ones(16))


def test_normalize_decoder_restores_unit_norm() -> None:
    """``normalize_decoder_`` is the post-step renormalisation that keeps the
    unit-norm constraint after the optimizer has moved ``W_dec`` off the sphere."""
    sae = SparseAutoEncoder(input_dim=8, hidden_dim=16, k=4, k_aux=4, mode="splade")
    sae.W_dec.data.mul_(3.7)
    assert not torch.allclose(sae.W_dec.norm(dim=-1), torch.ones(16), atol=1e-3)
    sae.normalize_decoder_()
    torch.testing.assert_close(sae.W_dec.norm(dim=-1), torch.ones(16))


def test_normalize_decoder_is_noop_in_csr_mode() -> None:
    """The method is safe to call in csr mode — it just does nothing rather than
    raising, so generic callbacks can fire it across every SAE in a model."""
    csr = SparseAutoEncoder(input_dim=8, hidden_dim=16, k=4, mode="csr")
    snapshot = {n: p.detach().clone() for n, p in csr.named_parameters()}
    csr.normalize_decoder_()
    for n, p in csr.named_parameters():
        torch.testing.assert_close(p, snapshot[n])


def test_parallel_gradient_component_is_stripped() -> None:
    """The backward hook on ``W_dec`` removes the gradient component parallel to
    each (unit) row. Combined with post-step renormalisation, training stays on
    the unit sphere."""
    import torch.nn.functional as F

    sae = SparseAutoEncoder(input_dim=8, hidden_dim=16, k=4, k_aux=4, mode="splade")
    sae.normalize_decoder_()  # ensure rows are unit
    out = sae({"token_embeddings": torch.randn(2, 4, 8)})
    out["decoded_token_embeddings_k"].sum().backward()

    rows_unit = F.normalize(sae.W_dec.detach(), dim=-1)
    parallel = (sae.W_dec.grad * rows_unit).sum(dim=-1)
    # Should be numerically zero — well below the magnitudes a real gradient would have.
    assert parallel.abs().max().item() < 1e-5


def test_corpus_normalization_buffers_only_when_splade_and_normalize() -> None:
    """The ``mean_bias`` / ``mean_norm`` buffers are splade-mode-only — they don't
    appear in csr mode, and they don't appear in splade mode with normalize=False."""
    s_norm = SparseAutoEncoder(input_dim=8, hidden_dim=16, k=4, k_aux=4, mode="splade", normalize=True)
    assert "mean_bias" in dict(s_norm.named_buffers())
    assert "mean_norm" in dict(s_norm.named_buffers())

    s_nonorm = SparseAutoEncoder(input_dim=8, hidden_dim=16, k=4, k_aux=4, mode="splade", normalize=False)
    assert "mean_bias" not in dict(s_nonorm.named_buffers())

    csr = SparseAutoEncoder(input_dim=8, hidden_dim=16, k=4, mode="csr", normalize=True)
    assert "mean_bias" not in dict(csr.named_buffers())


def test_init_corpus_normalization_mean_bias_is_chunking_invariant() -> None:
    """``mean_bias`` is a straight token-weighted streaming mean, so it must match
    exactly regardless of how the corpus is chunked.

    (``mean_norm`` is *not* chunking-invariant — it averages per-batch centered
    norms, matching the reference implementation, so we don't pin that here.)
    """
    sae_single = SparseAutoEncoder(input_dim=8, hidden_dim=16, k=4, k_aux=4, mode="splade", normalize=True)
    sae_chunked = SparseAutoEncoder(input_dim=8, hidden_dim=16, k=4, k_aux=4, mode="splade", normalize=True)

    torch.manual_seed(0)
    hidden = torch.randn(4, 50, 8) * 3.0 + 1.5
    mask = torch.ones(4, 50, dtype=torch.long)

    sae_single.init_corpus_normalization([(hidden, mask)])
    sae_chunked.init_corpus_normalization([(hidden[:1], mask[:1]), (hidden[1:3], mask[1:3]), (hidden[3:], mask[3:])])

    torch.testing.assert_close(sae_single.mean_bias, sae_chunked.mean_bias)


def test_init_corpus_normalization_rejects_csr_and_no_normalize() -> None:
    csr = SparseAutoEncoder(input_dim=8, hidden_dim=16, k=4, mode="csr")
    with pytest.raises(RuntimeError, match="splade"):
        csr.init_corpus_normalization([(torch.randn(1, 4, 8), None)])

    no_norm = SparseAutoEncoder(input_dim=8, hidden_dim=16, k=4, k_aux=4, mode="splade", normalize=False)
    with pytest.raises(RuntimeError, match="normalize"):
        no_norm.init_corpus_normalization([(torch.randn(1, 4, 8), None)])


def test_corpus_normalize_rescales_output_for_downstream_pooling() -> None:
    """When ``normalize=True``, the SAE input is on the normalised scale and the
    encoder's latents are correspondingly small; without rescaling the
    externally-written ``token_embeddings``, the downstream log1p would squash
    everything.

    With the rescale and zero biases, the input-divide and output-multiply by
    ``mean_norm`` cancel exactly — the SAE behaves identically to a no-normalize
    SAE with the same weights. Verify that, since it pins both the input
    normalization *and* the output rescale being applied (drop either and the
    cancellation breaks).
    """
    torch.manual_seed(0)
    sae_norm = SparseAutoEncoder(input_dim=8, hidden_dim=16, k=4, k_aux=4, mode="splade", normalize=True)
    sae_raw = SparseAutoEncoder(input_dim=8, hidden_dim=16, k=4, k_aux=4, mode="splade", normalize=False)
    # Copy the weights so only the normalization path differs.
    sae_raw.encoder.load_state_dict(sae_norm.encoder.state_dict())
    sae_raw.W_dec.data.copy_(sae_norm.W_dec.data)
    sae_raw.pre_bias.data.copy_(sae_norm.pre_bias.data)
    sae_raw.latent_bias.data.copy_(sae_norm.latent_bias.data)

    sae_norm.mean_norm.fill_(7.0)  # any nonzero scale
    sae_norm.eval()
    sae_raw.eval()

    x = torch.randn(2, 4, 8)
    with torch.inference_mode():
        out_norm = sae_norm({"token_embeddings": x.clone()})["token_embeddings"]
        out_raw = sae_raw({"token_embeddings": x.clone()})["token_embeddings"]

    torch.testing.assert_close(out_norm, out_raw)


def test_decode_denormalizes_in_splade_normalize_mode() -> None:
    """In splade+normalize, ``decode`` must undo the corpus normalisation so an
    SAE reconstruction loss can compare against the original-scale hidden state."""
    sae = SparseAutoEncoder(input_dim=8, hidden_dim=16, k=4, k_aux=4, mode="splade", normalize=True)
    sae.mean_bias.fill_(2.0)
    sae.mean_norm.fill_(5.0)
    sae.eval()
    # With latents = 0, decode collapses to `pre_bias * mean_norm + mean_bias`.
    # pre_bias is zero at init, so the result must be `mean_bias` (broadcast).
    zero_latents = torch.zeros(1, 16)
    decoded = sae.decode(zero_latents)
    torch.testing.assert_close(decoded, torch.full((1, 8), 2.0))


def test_splade_save_and_reload_round_trips_W_dec_and_corpus_stats(tmp_path) -> None:
    sae = SparseAutoEncoder(input_dim=8, hidden_dim=16, k=4, k_aux=4, mode="splade", normalize=True)
    sae.mean_bias.fill_(1.7)
    sae.mean_norm.fill_(3.3)
    sae.W_dec.data.normal_()
    sae.normalize_decoder_()

    sae.eval()
    x = torch.randn(2, 4, 8)
    with torch.inference_mode():
        before = sae({"token_embeddings": x.clone()})["token_embeddings"].clone()

    import tempfile as _tmp

    with _tmp.TemporaryDirectory(dir=tmp_path) as d:
        sae.save(d)
        reloaded = SparseAutoEncoder.load(d)

    assert torch.equal(sae.W_dec, reloaded.W_dec)
    assert torch.equal(sae.mean_bias, reloaded.mean_bias)
    assert torch.equal(sae.mean_norm, reloaded.mean_norm)
    with torch.inference_mode():
        after = reloaded({"token_embeddings": x.clone()})["token_embeddings"]
    torch.testing.assert_close(before, after)


def test_decoder_normalization_callback_fires_on_step_end() -> None:
    """The bundled callback walks the model and renormalises every splade-mode
    SAE on ``on_step_end``. Push the decoder off the sphere, simulate a step,
    and verify it's back."""
    from sentence_transformers.sparse_encoder.callbacks import SpladeDecoderNormalizationCallback

    sae = SparseAutoEncoder(input_dim=8, hidden_dim=16, k=4, k_aux=4, mode="splade")
    callback = SpladeDecoderNormalizationCallback(sae)

    sae.W_dec.data.mul_(2.5)
    assert not torch.allclose(sae.W_dec.norm(dim=-1), torch.ones(16), atol=1e-3)
    callback.on_step_end(args=None, state=None, control=None)
    torch.testing.assert_close(sae.W_dec.norm(dim=-1), torch.ones(16))
