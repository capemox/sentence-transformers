from __future__ import annotations

import tempfile

import pytest
import torch

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.modules import (
    MLMTransform,
    SparseAutoEncoder,
    SpladePooling,
    Transformer,
)


def test_forward_preserves_shape() -> None:
    head = MLMTransform(hidden_size=32)
    x = torch.randn(2, 5, 32)
    out = head({"token_embeddings": x})
    assert tuple(out["token_embeddings"].shape) == (2, 5, 32)


def test_from_mlm_model_bert_matches_internal_transform() -> None:
    """The whole point of this module is to be bit-equal to the MLM head's transform
    sub-block. If that ever drifts the paper's recipe breaks silently — so pin it."""
    from transformers import BertConfig, BertForMaskedLM

    cfg = BertConfig(vocab_size=50, hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64)
    bert = BertForMaskedLM(cfg).eval()
    head = MLMTransform.from_mlm_model(bert)

    hidden = torch.randn(1, 4, 32)
    with torch.no_grad():
        expected = bert.cls.predictions.transform(hidden)
        got = head({"token_embeddings": hidden})["token_embeddings"]
    torch.testing.assert_close(got, expected)


def test_from_mlm_model_distilbert_matches_internal_transform() -> None:
    from transformers import DistilBertConfig, DistilBertForMaskedLM

    cfg = DistilBertConfig(vocab_size=50, dim=24, n_layers=1, n_heads=2, hidden_dim=48)
    distil = DistilBertForMaskedLM(cfg).eval()
    head = MLMTransform.from_mlm_model(distil)

    hidden = torch.randn(1, 4, 24)
    with torch.no_grad():
        # DistilBERT's MLM head, expanded.
        expected = distil.vocab_layer_norm(distil.activation(distil.vocab_transform(hidden)))
        got = head({"token_embeddings": hidden})["token_embeddings"]
    torch.testing.assert_close(got, expected)


def test_from_mlm_model_rejects_unknown_layout() -> None:
    """If a new MLM family with an unfamiliar head shows up, we must fail loudly
    rather than silently picking up the wrong sub-block."""

    class FakeMLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("C", (), {"hidden_act": "gelu"})()
            self.encoder = torch.nn.Linear(8, 8)  # no MLM head at all

    with pytest.raises(NotImplementedError, match="MLM transform"):
        MLMTransform.from_mlm_model(FakeMLM())


def test_save_and_load_round_trip() -> None:
    head = MLMTransform(hidden_size=16, hidden_act="gelu", layer_norm_eps=1e-5)
    hidden = torch.randn(1, 3, 16)
    before = head({"token_embeddings": hidden.clone()})["token_embeddings"]

    with tempfile.TemporaryDirectory() as out:
        head.save(out)
        reloaded = MLMTransform.load(out)

    after = reloaded({"token_embeddings": hidden.clone()})["token_embeddings"]
    torch.testing.assert_close(before, after)
    # config_keys also round-trip.
    assert reloaded.hidden_size == 16
    assert reloaded.hidden_act == "gelu"
    assert reloaded.layer_norm_eps == 1e-5


def test_full_sae_splade_pipeline_composes() -> None:
    """End-to-end: Transformer(feature-extraction) -> MLMTransform -> SparseAutoEncoder(splade) -> SpladePooling.

    Builds the paper-faithful pipeline on a tiny BERT and checks the aggregated embedding
    is sparse + has the SAE's hidden_dim. The MLMTransform is constructed from random
    weights here (since the tiny test backbone isn't an MLM checkpoint); the *shape* /
    *plumbing* contract is what we're pinning, not the trained values.
    """
    backbone = Transformer(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors",
        transformer_task="feature-extraction",
    )
    hidden = backbone.get_embedding_dimension()
    head = MLMTransform(hidden_size=hidden)
    sae = SparseAutoEncoder(input_dim=hidden, hidden_dim=64, k=4, k_aux=2, mode="splade")
    pool = SpladePooling(pooling_strategy="max")

    model = SparseEncoder(modules=[backbone, head, sae, pool])
    emb = model.encode(["hello world", "this is a longer test sentence"])
    dense = emb.to_dense() if emb.is_sparse else emb
    assert tuple(dense.shape) == (2, 64)
    assert (dense >= 0).all(), "post log1p the embedding is non-negative"
    # The MLMTransform doesn't kill sparsity — there should still be many zeros.
    assert int((dense > 0).sum()) < dense.numel()
