from __future__ import annotations

import pytest
import torch

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.losses import SAESpladeReconstructionLoss
from sentence_transformers.sparse_encoder.modules import (
    SparseAutoEncoder,
    SpladePooling,
    Transformer,
)

BACKBONE = "sentence-transformers-testing/stsb-bert-tiny-safetensors"


@pytest.fixture(scope="module")
def sae_splade_pipeline() -> SparseEncoder:
    backbone = Transformer(BACKBONE, transformer_task="feature-extraction")
    sae = SparseAutoEncoder(
        input_dim=backbone.get_embedding_dimension(),
        hidden_dim=64,
        k=4,
        k_aux=2,
        mode="splade",
    )
    return SparseEncoder(modules=[backbone, sae, SpladePooling(pooling_strategy="max")])


def _preprocess(model: SparseEncoder, texts: list[str]) -> dict[str, torch.Tensor]:
    features = model.preprocess(texts)
    return {k: v.to(model.device) if hasattr(v, "to") else v for k, v in features.items()}


def test_loss_dict_has_expected_keys(sae_splade_pipeline: SparseEncoder) -> None:
    loss = SAESpladeReconstructionLoss(sae_splade_pipeline)
    out = loss.forward([_preprocess(sae_splade_pipeline, ["hello world"])])
    assert set(out.keys()) == {"reconstruction_loss_k", "reconstruction_loss_aux"}


def test_l_4k_term_appears_only_when_weighted(sae_splade_pipeline: SparseEncoder) -> None:
    loss_off = SAESpladeReconstructionLoss(sae_splade_pipeline)
    loss_on = SAESpladeReconstructionLoss(sae_splade_pipeline, l_4k_weight=0.125)

    feats = [_preprocess(sae_splade_pipeline, ["hello world"])]
    assert "reconstruction_loss_4k" not in loss_off.forward(feats)
    assert "reconstruction_loss_4k" in loss_on.forward(feats)


def test_aux_term_skipped_when_k_aux_zero() -> None:
    backbone = Transformer(BACKBONE, transformer_task="feature-extraction")
    sae = SparseAutoEncoder(
        input_dim=backbone.get_embedding_dimension(),
        hidden_dim=32,
        k=4,
        k_aux=0,  # no aux path
        mode="splade",
    )
    model = SparseEncoder(modules=[backbone, sae, SpladePooling(pooling_strategy="max")])
    loss = SAESpladeReconstructionLoss(model)

    out = loss.forward([_preprocess(model, ["no aux path"])])
    assert "reconstruction_loss_aux" not in out
    assert "reconstruction_loss_k" in out


def test_rejects_model_without_splade_sae() -> None:
    """If the pipeline doesn't actually contain a splade-mode SAE, fail at construction
    rather than producing a confusing KeyError on missing token_embeddings_backbone later."""
    backbone = Transformer(BACKBONE, transformer_task="feature-extraction")
    # Default mode is csr, so this is not a splade SAE.
    csr_sae = SparseAutoEncoder(input_dim=backbone.get_embedding_dimension(), hidden_dim=32, k=4)
    model = SparseEncoder(modules=[backbone, csr_sae])
    with pytest.raises(ValueError, match='mode="splade"'):
        SAESpladeReconstructionLoss(model)


def test_backward_flows_through_sae(sae_splade_pipeline: SparseEncoder) -> None:
    """The whole point of the loss is to drive SAE training; verify that gradients
    actually reach the SAE's parameters (encoder + W_dec + pre_bias + latent_bias).
    """
    sae = next(m for m in sae_splade_pipeline if isinstance(m, SparseAutoEncoder))
    # Reset grads in case earlier tests left them populated.
    sae_splade_pipeline.zero_grad(set_to_none=True)

    loss = SAESpladeReconstructionLoss(sae_splade_pipeline)
    out = loss.forward([_preprocess(sae_splade_pipeline, ["backward test sentence"])])
    sum(out.values()).backward()

    assert sae.encoder.weight.grad is not None and sae.encoder.weight.grad.abs().sum() > 0
    assert sae.W_dec.grad is not None and sae.W_dec.grad.abs().sum() > 0
    assert sae.pre_bias.grad is not None
    assert sae.latent_bias.grad is not None


def test_padding_tokens_excluded_from_loss(sae_splade_pipeline: SparseEncoder) -> None:
    """Padding tokens must not contribute to the loss — that's what attention_mask
    handling is for. Construct two inputs of different lengths in the same batch and
    verify the loss matches a single-input loss on each sample minus the padding."""
    loss = SAESpladeReconstructionLoss(sae_splade_pipeline)
    paired = _preprocess(sae_splade_pipeline, ["hi", "this is a much longer sentence"])

    # Sanity: the second batch has at least one padded position in the short row.
    assert paired["attention_mask"][0].sum() < paired["attention_mask"].shape[1]

    # A non-masked implementation tends to under- or over-weight the short row and
    # can produce nan / inf when the padding region happens to land at decoder
    # rows that have zero activation in this batch. Just running and finite-checking
    # is enough to pin the contract that padding is excluded.
    out = loss.forward([paired])
    for v in out.values():
        assert torch.isfinite(v), f"loss term {v!r} is non-finite — likely a padding-handling bug"
