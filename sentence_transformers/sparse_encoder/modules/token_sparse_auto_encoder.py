from __future__ import annotations

import torch

from sentence_transformers.sparse_encoder.modules.sparse_auto_encoder import SparseAutoEncoder


class TokenSparseAutoEncoder(SparseAutoEncoder):
    """
    Per-token variant of :class:`SparseAutoEncoder`.

    Same math (pre-bias, tied decoder, top-K, AuxK, dead-neuron stats) — only the input
    contract differs: it reads ``features["token_embeddings"]`` of shape
    ``(batch, seq_length, input_dim)`` and writes top-K SAE latents of shape
    ``(batch, seq_length, hidden_dim)`` back to the same key, leaving the result ready
    for any per-token downstream module.

    Training-mode forward also exposes the same family of intermediates as
    :class:`SparseAutoEncoder`, prefixed ``token_embeddings_*`` /
    ``decoded_token_embeddings_*``, so token-level SAE reconstruction or AuxK losses can
    be written against them.

    Typical use case is the SAE-SPLADE pipeline from `From Tokens to Concepts: Leveraging
    SAE for SPLADE <https://huggingface.co/papers/2604.21511>`_:

    .. code-block:: python

        from sentence_transformers import SparseEncoder
        from sentence_transformers.sparse_encoder.modules import (
            SpladePooling,
            TokenSparseAutoEncoder,
            Transformer,
        )

        transformer = Transformer("distilbert/distilbert-base-uncased", transformer_task="feature-extraction")
        sae = TokenSparseAutoEncoder(input_dim=transformer.get_embedding_dimension(), hidden_dim=65536, k=0)
        model = SparseEncoder(modules=[transformer, sae, SpladePooling(pooling_strategy="max")])

    Setting ``k=0`` (or any value ``>= hidden_dim``) skips the per-token top-K mask; the
    downstream :class:`SpladePooling`'s ReLU + log1p + max-pool is then the only source
    of sparsity, matching the SPLADE fine-tuning phase of the paper.

    All constructor arguments are inherited from :class:`SparseAutoEncoder`.
    """

    forward_kwargs = {"max_active_dims"}

    def _topk_disabled(self, k: int) -> bool:
        return k <= 0 or k >= self.hidden_dim

    def _topk_or_full(self, latents_pre_act: torch.Tensor, k: int, *, compute_aux: bool):
        # Either run :meth:`SparseAutoEncoder.top_k` or skip masking entirely.
        if self._topk_disabled(k):
            return torch.relu(latents_pre_act), None
        return self.top_k(latents_pre_act, k, compute_aux=compute_aux)

    def forward(
        self, features: dict[str, torch.Tensor], max_active_dims: int | None = None
    ) -> dict[str, torch.Tensor]:
        k = max_active_dims if max_active_dims is not None else self.k
        x = features["token_embeddings"]  # (batch, seq, input_dim)

        x_prep, info = self.prepare(x)
        latents_pre_act = self.encode_pre_act(x_prep)

        if torch.is_inference_mode_enabled():
            latents_k, _ = self._topk_or_full(latents_pre_act, k, compute_aux=False)
            features["token_embeddings"] = latents_k
            return features

        latents_k, latents_auxk = self._topk_or_full(latents_pre_act, k, compute_aux=True)
        if self._topk_disabled(k):
            latents_4k = latents_k
        else:
            latents_4k, _ = self.top_k(latents_pre_act, min(4 * k, self.hidden_dim))

        recons_k = self.decode(latents_k, info)
        recons_4k = self.decode(latents_4k, info)
        recons_aux = self.decode(latents_auxk, info) if latents_auxk is not None else None

        features.update(
            {
                "token_embeddings_backbone": x_prep,
                "token_embeddings_encoded": latents_pre_act,
                "token_embeddings_encoded_4k": latents_4k,
                "auxiliary_token_embeddings": latents_auxk,
                "decoded_token_embeddings_k": recons_k,
                "decoded_token_embeddings_4k": recons_4k,
                "decoded_token_embeddings_aux": recons_aux,
                "decoded_token_embeddings_k_pre_bias": recons_k - self.pre_bias,
            }
        )
        features["token_embeddings"] = latents_k
        return features

    def __repr__(self) -> str:
        return f"TokenSparseAutoEncoder({self.get_config_dict()})"

    def get_embedding_dimension(self) -> int:
        """Per-token output width (i.e. the eventual aggregated vector size)."""
        return self.hidden_dim
