from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from sentence_transformers.sparse_encoder.losses.csr import normalized_mean_squared_error
from sentence_transformers.sparse_encoder.model import SparseEncoder
from sentence_transformers.sparse_encoder.modules.sparse_auto_encoder import SparseAutoEncoder


class SAESpladeReconstructionLoss(nn.Module):
    """SAE pretraining reconstruction loss for ``mode="splade"`` :class:`SparseAutoEncoder`
    modules.

    Mirrors :class:`CSRReconstructionLoss`, but reads the token-level training
    intermediates produced by a splade-mode SAE
    (``token_embeddings_backbone`` / ``decoded_token_embeddings_*``) and masks out padding
    via ``attention_mask`` so only valid tokens contribute to the loss. Designed for
    *phase 1* of the SAE-SPLADE recipe: train the SAE on a frozen backbone with no
    contrastive component; once it's pretrained, swap in :class:`SpladeLoss` to fine-tune.

    The loss has two parts, both standard for top-K SAEs and matching the paper's
    ``SAEReconstructionRegu`` (``sae_splade/src/sae/hooks.py``):

    * ``L_k`` — MSE between the original backbone hidden state and the top-K
      reconstruction, averaged over valid tokens.
    * ``L_aux`` — normalised MSE between the AuxK reconstruction and the residual
      ``x - decoder(top_k_latents).detach()``, used to keep dead latents from staying
      dead. Skipped when ``k_aux=0`` on the SAE.

    An optional Matryoshka-style ``L_4k`` (MSE against the top-4k reconstruction) is
    available too, but defaults to off — it's not in the paper's recipe.

    Args:
        model: :class:`SparseEncoder` containing at least one
            ``SparseAutoEncoder(mode="splade")`` module.
        beta: weight for ``L_aux`` in the returned loss dict. Defaults to ``1/32`` —
            the same scale the paper uses (``coeff_aux`` is roughly an order of
            magnitude smaller than the main coefficient).
        l_4k_weight: weight for the optional ``L_4k`` Matryoshka term. Defaults to
            ``0.0`` (term omitted from the returned dict). Set to ``1/8`` to match
            :class:`CSRReconstructionLoss`.

    Example:
        ::

            from sentence_transformers import SparseEncoder
            from sentence_transformers.sparse_encoder.modules import (
                MLMTransform,
                SparseAutoEncoder,
                SpladePooling,
                Transformer,
            )
            from sentence_transformers.sparse_encoder.losses import SAESpladeReconstructionLoss
            from sentence_transformers.sparse_encoder.callbacks import (
                SpladeDecoderNormalizationCallback,
            )

            backbone = Transformer("distilbert/distilbert-base-uncased", transformer_task="feature-extraction")
            head = MLMTransform.from_pretrained_mlm("distilbert/distilbert-base-uncased")
            sae = SparseAutoEncoder(
                input_dim=backbone.get_embedding_dimension(),
                hidden_dim=65536,
                k=128,            # tight bottleneck during SAE pretraining
                k_aux=512,
                mode="splade",
                normalize=True,   # set mean_bias/mean_norm via init_corpus_normalization() first
            )
            # Freeze backbone + MLM transform; train only the SAE.
            backbone.requires_grad_(False)
            head.requires_grad_(False)

            model = SparseEncoder(modules=[backbone, head, sae, SpladePooling(pooling_strategy="max")])
            loss = SAESpladeReconstructionLoss(model)

            # ... build a single-column dataset of texts, then ...
            # trainer = SparseEncoderTrainer(model, ..., loss=loss, callbacks=[SpladeDecoderNormalizationCallback(model)])
    """

    def __init__(self, model: SparseEncoder, beta: float = 1 / 32, l_4k_weight: float = 0.0) -> None:
        super().__init__()
        if not any(isinstance(m, SparseAutoEncoder) and m.mode == "splade" for m in model.modules()):
            raise ValueError(
                "SAESpladeReconstructionLoss expects the model to contain at least one "
                'SparseAutoEncoder configured with mode="splade".'
            )
        self.model = model
        self.beta = beta
        self.l_4k_weight = l_4k_weight

    def forward(
        self, sentence_features: Iterable[dict[str, torch.Tensor]], labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        outputs = [self.model(sentence_feature) for sentence_feature in sentence_features]
        return self.compute_loss_from_embeddings(outputs)

    def compute_loss_from_embeddings(self, outputs: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Aggregate the per-column reconstruction losses.

        For a single-column dataset (typical SAE pretraining setup with one text per
        row), this loops once and returns the per-token losses. For multi-column
        datasets the losses are averaged across columns, matching
        :class:`CSRReconstructionLoss`.
        """
        total_L_k = 0.0
        total_L_4k = 0.0
        total_L_aux = 0.0
        had_aux = False
        had_4k = False

        for features in outputs:
            x = features["token_embeddings_backbone"]
            recons_k = features["decoded_token_embeddings_k"]
            recons_aux = features.get("auxiliary_token_embeddings")
            recons_aux_decoded = features.get("decoded_token_embeddings_aux")
            recons_4k = features.get("decoded_token_embeddings_4k")
            reconsk_pre_bias = features["decoded_token_embeddings_k_pre_bias"]
            attention_mask = features.get("attention_mask")

            # L_k: per-valid-token MSE on the primary reconstruction.
            total_L_k = total_L_k + self._masked_mse(recons_k, x, attention_mask)

            # L_aux: same structure as CSRReconstructionLoss — predict the residual
            # ``x - recons_k_pre_bias.detach()`` using the AuxK reconstruction.
            if recons_aux is not None and recons_aux_decoded is not None:
                aux_target = x - reconsk_pre_bias.detach()
                total_L_aux = total_L_aux + self._masked_normalized_mse(recons_aux_decoded, aux_target, attention_mask)
                had_aux = True

            # Optional Matryoshka L_4k.
            if self.l_4k_weight > 0 and recons_4k is not None:
                total_L_4k = total_L_4k + self._masked_mse(recons_4k, x, attention_mask)
                had_4k = True

        n = max(len(outputs), 1)
        loss_dict: dict[str, torch.Tensor] = {"reconstruction_loss_k": total_L_k / n}
        if had_aux:
            loss_dict["reconstruction_loss_aux"] = self.beta * total_L_aux / n
        if had_4k:
            loss_dict["reconstruction_loss_4k"] = self.l_4k_weight * total_L_4k / n
        return loss_dict

    def get_config_dict(self) -> dict:
        return {"beta": self.beta, "l_4k_weight": self.l_4k_weight}

    @property
    def citation(self) -> str:
        return """
@article{zong2026tokens,
    title={From Tokens to Concepts: Leveraging SAE for SPLADE},
    author={Zong, Yuxuan and Vast, Mathias and Van Cooten, Basile and Soulier, Laure and Piwowarski, Benjamin},
    journal={arXiv preprint arXiv:2604.21511},
    year={2026}
}
"""

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _masked_mse(
        prediction: torch.Tensor, target: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if attention_mask is None:
            return F.mse_loss(prediction, target)
        valid = attention_mask.bool()
        return F.mse_loss(prediction[valid], target[valid])

    @staticmethod
    def _masked_normalized_mse(
        prediction: torch.Tensor, target: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if attention_mask is None:
            return normalized_mean_squared_error(prediction, target)
        valid = attention_mask.bool()
        return normalized_mean_squared_error(prediction[valid], target[valid])
