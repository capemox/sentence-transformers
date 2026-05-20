from __future__ import annotations

import logging
import os

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import torch
import torch.nn as nn

from sentence_transformers.base.modules.module import Module

logger = logging.getLogger(__name__)


def _resolve_activation(name: str) -> nn.Module:
    """Map an MLM-head activation name to an instantiated ``nn.Module``.

    Uses ``transformers.activations.ACT2FN`` so the activation matches every
    BERT-family MLM head bit-for-bit (they use the same registry internally).
    """
    from transformers.activations import ACT2FN

    activation = ACT2FN[name]
    if not isinstance(activation, nn.Module):
        raise TypeError(f"Activation {name!r} did not resolve to an nn.Module (got {type(activation).__name__})")
    return activation


def _find_mlm_transform_layers(mlm_model: nn.Module) -> tuple[nn.Linear, nn.LayerNorm, str]:
    """Locate ``(Linear, LayerNorm, hidden_act_name)`` inside a BERT-family MLM head.

    The transform sub-block is named slightly differently in each architecture, so we
    sniff for the known layouts. Extend this when adding support for a new MLM family.
    """
    config = getattr(mlm_model, "config", None)
    cfg_act = getattr(config, "hidden_act", None) or getattr(config, "hidden_activation", None) or "gelu"

    # BERT-family (BERT, ERNIE, RetriBERT, ...): cls.predictions.transform.{dense,LayerNorm}
    cls_attr = getattr(mlm_model, "cls", None)
    if cls_attr is not None and hasattr(cls_attr, "predictions") and hasattr(cls_attr.predictions, "transform"):
        t = cls_attr.predictions.transform
        return t.dense, t.LayerNorm, cfg_act

    # DistilBERT: top-level vocab_transform / vocab_layer_norm
    if hasattr(mlm_model, "vocab_transform") and hasattr(mlm_model, "vocab_layer_norm"):
        return mlm_model.vocab_transform, mlm_model.vocab_layer_norm, "gelu"

    # RoBERTa / XLM-R / CamemBERT: lm_head.{dense,layer_norm}
    if (
        hasattr(mlm_model, "lm_head")
        and hasattr(mlm_model.lm_head, "dense")
        and hasattr(mlm_model.lm_head, "layer_norm")
    ):
        return mlm_model.lm_head.dense, mlm_model.lm_head.layer_norm, "gelu"

    # ModernBERT: head.{dense,norm}
    if hasattr(mlm_model, "head") and hasattr(mlm_model.head, "dense") and hasattr(mlm_model.head, "norm"):
        return mlm_model.head.dense, mlm_model.head.norm, cfg_act

    raise NotImplementedError(
        f"Don't know how to locate the MLM transform sub-block inside {type(mlm_model).__name__}. "
        "Supported architectures: BertForMaskedLM, DistilBertForMaskedLM, RobertaForMaskedLM, "
        "ModernBertForMaskedLM (and close siblings). Construct MLMTransform manually and copy "
        "the corresponding `Linear` / `LayerNorm` weights to extend support."
    )


class MLMTransform(Module):
    """The pre-projection block of a BERT-family Masked-LM head: a
    ``Linear → activation → LayerNorm`` stack that sits between the encoder's last
    hidden state and the final vocabulary projection.

    Used between a :class:`Transformer` and a :class:`SparseAutoEncoder` in SAE-SPLADE
    pipelines so the SAE consumes the representation BERT was actually trained to use
    immediately before predicting words — the recipe from
    `From Tokens to Concepts: Leveraging SAE for SPLADE
    <https://huggingface.co/papers/2604.21511>`_ — rather than the raw
    ``last_hidden_state`` (which is one layer earlier) or the full vocab logits (which
    are vocab-sized).

    Construct freshly to train from scratch, or use :meth:`from_pretrained_mlm` to copy
    just the transform sub-block weights out of an existing MLM checkpoint.

    Example:

    .. code-block:: python

        from sentence_transformers import SparseEncoder
        from sentence_transformers.sparse_encoder.modules import (
            MLMTransform,
            SparseAutoEncoder,
            SpladePooling,
            Transformer,
        )

        backbone = Transformer("distilbert/distilbert-base-uncased", transformer_task="feature-extraction")
        head = MLMTransform.from_pretrained_mlm("distilbert/distilbert-base-uncased")
        sae = SparseAutoEncoder(
            input_dim=backbone.get_embedding_dimension(),
            hidden_dim=65536,
            k=0,
            mode="splade",
        )
        model = SparseEncoder(modules=[backbone, head, sae, SpladePooling(pooling_strategy="max")])

    Args:
        hidden_size: Dimensionality of the input / output hidden state (the backbone's
            hidden size; preserved by the transform).
        hidden_act: Name of the activation between the Linear and the LayerNorm. ``"gelu"``
            matches every BERT-family MLM head ships with.
        layer_norm_eps: ``eps`` of the LayerNorm. ``1e-12`` is BERT's default.
    """

    config_keys: list[str] = ["hidden_size", "hidden_act", "layer_norm_eps"]

    def __init__(
        self,
        hidden_size: int,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = _resolve_activation(hidden_act)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = features["token_embeddings"]
        features["token_embeddings"] = self.layer_norm(self.activation(self.dense(x)))
        return features

    def get_embedding_dimension(self) -> int:
        return self.hidden_size

    def __repr__(self) -> str:
        return f"MLMTransform({self.get_config_dict()})"

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        self.save_config(output_path)
        self.save_torch_weights(output_path, safe_serialization=safe_serialization)

    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        **kwargs,
    ) -> Self:
        hub_kwargs = {
            "subfolder": subfolder,
            "token": token,
            "cache_folder": cache_folder,
            "revision": revision,
            "local_files_only": local_files_only,
        }
        config = cls.load_config(model_name_or_path=model_name_or_path, **hub_kwargs)
        model = cls(**config)
        model = cls.load_torch_weights(model_name_or_path=model_name_or_path, model=model, **hub_kwargs)
        return model

    @classmethod
    def from_mlm_model(cls, mlm_model: nn.Module) -> Self:
        """Build an :class:`MLMTransform` whose weights are copied from the MLM
        head of an already-loaded ``AutoModelForMaskedLM``. Useful for testing or
        when the MLM model is already in memory."""
        dense, layer_norm, hidden_act = _find_mlm_transform_layers(mlm_model)
        instance = cls(
            hidden_size=dense.in_features,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm.eps,
        )
        instance.dense.load_state_dict(dense.state_dict())
        instance.layer_norm.load_state_dict(layer_norm.state_dict())
        return instance

    @classmethod
    def from_pretrained_mlm(
        cls,
        model_name_or_path: str | os.PathLike,
        cache_dir: str | None = None,
        **kwargs,
    ) -> Self:
        """Construct an :class:`MLMTransform` by copying the transform-sub-block
        weights from a pretrained MLM checkpoint.

        The full MLM model is loaded once to harvest the weights, then discarded —
        only the ``Linear`` and ``LayerNorm`` are retained in the returned module.
        Supports BERT-family checkpoints (BERT, DistilBERT, RoBERTa, ModernBERT,
        and close siblings); falls back with a clear error for unsupported layouts.

        Args:
            model_name_or_path: HF Hub id or local path of a pretrained MLM checkpoint.
            cache_dir: Optional HF cache directory.
            **kwargs: Forwarded to ``AutoModelForMaskedLM.from_pretrained``.
        """
        from transformers import AutoModelForMaskedLM

        mlm = AutoModelForMaskedLM.from_pretrained(model_name_or_path, cache_dir=cache_dir, **kwargs)
        return cls.from_mlm_model(mlm)
