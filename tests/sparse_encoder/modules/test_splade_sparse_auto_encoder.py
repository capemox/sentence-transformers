from __future__ import annotations

import tempfile
from contextlib import nullcontext

import pytest
import torch

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.modules import (
    SpladePooling,
    SpladeSparseAutoEncoder,
    Transformer,
)

BACKBONE = "sentence-transformers-testing/stsb-bert-tiny-safetensors"


@pytest.fixture(scope="module")
def sae_splade_model() -> SparseEncoder:
    """Build the canonical SAE-SPLADE pipeline on a tiny BERT.

    ``Transformer(feature-extraction)`` -> :class:`SpladeSparseAutoEncoder` -> :class:`SpladePooling`.
    """
    transformer = Transformer(BACKBONE, transformer_task="feature-extraction")
    sae = SpladeSparseAutoEncoder(
        input_dim=transformer.get_embedding_dimension(),
        hidden_dim=64,
        k=4,
        k_aux=2,
        dead_threshold=2,
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


def test_subclass_relationship() -> None:
    from sentence_transformers.sparse_encoder.modules import SparseAutoEncoder

    # We deliberately subclass `SparseAutoEncoder` so the SAE math (pre-bias, tied decoder,
    # top-K, AuxK, dead-neuron stats) is shared, not duplicated.
    assert issubclass(SpladeSparseAutoEncoder, SparseAutoEncoder)


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
def test_training_intermediates_exposed(
    sae_splade_model: SparseEncoder, is_inference: bool, expected_extra_keys: set
) -> None:
    """In training mode the SAE must expose the same family of intermediates as
    :class:`SparseAutoEncoder` (just at the token level), so token-level SAE
    reconstruction losses can be written against them."""
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
        # Decoded tensors round-trip back to the backbone hidden size.
        assert tuple(out["decoded_token_embeddings_k"].shape) == (1, seq_len, hidden)
        # AuxK has its own k_aux trailing dimension, dense over hidden_dim like top_k.
        assert out["auxiliary_token_embeddings"].shape[-1] == sae_splade_model[1].hidden_dim


def test_k_zero_disables_topk_mask() -> None:
    """``k=0`` is the SPLADE fine-tuning path: no top-K mask, sparsity comes from
    ReLU + log1p + max-pool only. Verify it (a) runs and (b) skips the AuxK output."""
    transformer = Transformer(BACKBONE, transformer_task="feature-extraction")
    sae = SpladeSparseAutoEncoder(
        input_dim=transformer.get_embedding_dimension(),
        hidden_dim=32,
        k=0,
        k_aux=4,
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

    # token_embeddings is the SAE output before pooling: count active latents per token.
    per_token_nnz_default = (default["token_embeddings"] > 0).sum(dim=-1)
    per_token_nnz_tighter = (tighter["token_embeddings"] > 0).sum(dim=-1)
    assert torch.all(per_token_nnz_default <= sae_splade_model[1].k)
    assert torch.all(per_token_nnz_tighter <= 2)
    # And a tighter top-K can never enlarge the pooled vector's support.
    assert int((tighter["sentence_embedding"] > 0).sum()) <= int((default["sentence_embedding"] > 0).sum())


def test_save_and_reload(sae_splade_model: SparseEncoder, tmp_path) -> None:
    inputs = ["save reload check"]
    before = _to_dense(sae_splade_model.encode(inputs))

    with tempfile.TemporaryDirectory(dir=tmp_path) as out:
        sae_splade_model.save_pretrained(out)
        reloaded = SparseEncoder(out)

    assert any(isinstance(m, SpladeSparseAutoEncoder) for m in reloaded), (
        "Reloaded modules.json must round-trip the SpladeSparseAutoEncoder class"
    )
    after = _to_dense(reloaded.encode(inputs))
    torch.testing.assert_close(before, after)


def test_dead_neuron_stats_update(sae_splade_model: SparseEncoder) -> None:
    """``stats_last_nonzero`` is the buffer the parent ``SparseAutoEncoder``'s AuxK mask
    consults to find dead latents. Each training-mode forward must update it; the values
    must keep moving so dead latents can rotate in and out of the AuxK pool."""
    sae = sae_splade_model[1]
    assert isinstance(sae, SpladeSparseAutoEncoder)
    sae.stats_last_nonzero.zero_()

    inputs = sae_splade_model.preprocess(["dead neuron stats"])
    inputs = {k: v.to(sae_splade_model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    sae_splade_model(inputs)
    after_one = sae.stats_last_nonzero.clone()
    assert (after_one > 0).all(), "every latent's step counter must advance on each forward"

    sae_splade_model(inputs)
    assert not torch.equal(sae.stats_last_nonzero, after_one), "stats buffer must keep moving"
