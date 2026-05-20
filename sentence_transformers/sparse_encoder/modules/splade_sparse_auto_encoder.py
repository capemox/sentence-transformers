from __future__ import annotations

import torch

from sentence_transformers.sparse_encoder.modules.sparse_auto_encoder import SparseAutoEncoder


class SpladeSparseAutoEncoder(SparseAutoEncoder):
    """
    Token-level Sparse Autoencoder for SAE-SPLADE pipelines.

    This is the module proposed in `From Tokens to Concepts: Leveraging SAE for SPLADE
    <https://huggingface.co/papers/2604.21511>`_. It replaces SPLADE's vocabulary projection
    with a Sparse Autoencoder learned on top of the backbone's per-token hidden states,
    so that the downstream :class:`~sentence_transformers.sparse_encoder.modules.SpladePooling`
    aggregates over a learned dictionary of `hidden_dim` *concepts* instead of the
    backbone's wordpiece vocabulary.

    The math is identical to :class:`SparseAutoEncoder` (pre-bias, tied decoder, top-K,
    auxiliary top-K, dead-neuron tracking); only the contract differs:

    * Input: ``features["token_embeddings"]`` of shape ``(batch, seq_length, input_dim)`` —
      the backbone's last hidden states (use ``Transformer(transformer_task="feature-extraction")``).
    * Output (inference): ``features["token_embeddings"]`` of shape
      ``(batch, seq_length, hidden_dim)`` — top-K SAE latents per token, ready to be
      max-pooled by :class:`SpladePooling`.
    * Output (training): the same plus a set of ``token_embeddings_*`` / ``decoded_token_embeddings_*``
      tensors mirroring the training intermediates of :class:`SparseAutoEncoder`, available
      for any token-level SAE reconstruction / auxiliary loss.

    Recommended pipeline:

    .. code-block:: python

        from sentence_transformers import SparseEncoder
        from sentence_transformers.sparse_encoder.modules import (
            SpladePooling,
            SpladeSparseAutoEncoder,
            Transformer,
        )

        transformer = Transformer("distilbert/distilbert-base-uncased", transformer_task="feature-extraction")
        sae = SpladeSparseAutoEncoder(
            input_dim=transformer.get_embedding_dimension(),
            hidden_dim=65536,
            k=0,  # 0 = no top-K mask at SPLADE-finetuning time
        )
        model = SparseEncoder(modules=[transformer, sae, SpladePooling(pooling_strategy="max")])

    Args (inherited from :class:`SparseAutoEncoder`):
        input_dim: Dimension of the backbone hidden states.
        hidden_dim: Width of the SAE latent dictionary (the ``sae_width`` from the paper).
            Defaults to 512 (the same default as :class:`SparseAutoEncoder`); the paper uses
            values up to ``2**17``.
        k: Top-K kept per token. Set to ``0`` (or any value ``>= hidden_dim``) to disable
            the top-K mask entirely, in which case sparsity is produced solely by ReLU +
            log1p + max-pool in the downstream :class:`SpladePooling`. Defaults to 8.
        k_aux: Top-K over dead-neuron-masked pre-activations for the auxiliary
            reconstruction loss. Set to 0 to skip the auxiliary path. Defaults to 512.
        normalize: Whether to per-token layer-normalize the input before encoding.
        dead_threshold: Steps of inactivity after which a latent is considered dead.
    """

    # Inherited config_keys / save / load are sufficient — no extra persisted state.

    forward_kwargs = {"max_active_dims"}

    def _disable_topk(self, k: int) -> bool:
        """A non-positive ``k``, or one as wide as the dictionary, means "no top-K mask"."""
        return k <= 0 or k >= self.hidden_dim

    def _topk_or_full(self, latents_pre_act: torch.Tensor, k: int, *, compute_aux: bool):
        """Either run :meth:`SparseAutoEncoder.top_k` or skip the masking entirely.

        The full-dictionary path is what the paper uses during SPLADE fine-tuning: the
        SAE's ReLU + the downstream ``log1p`` + max-pool are enough to drive sparsity,
        and the top-K constraint is dropped to let SPLADE pick concepts freely.
        """
        if self._disable_topk(k):
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
        # Matryoshka-style top-4k slice, matching the sentence-level SparseAutoEncoder.
        # Skipped when the main path already uses the full dictionary.
        if self._disable_topk(k):
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
        return f"SpladeSparseAutoEncoder({self.get_config_dict()})"

    def get_embedding_dimension(self) -> int:
        """Width of the per-token SAE output (i.e. the eventual SPLADE vector size)."""
        return self.hidden_dim
