from __future__ import annotations

from typing import Literal

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import torch
import torch.nn as nn
import torch.nn.functional as F

from sentence_transformers.base.modules.module import Module

SparseAutoEncoderMode = Literal["csr", "splade"]


class TiedTranspose(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.linear.bias is not None:
            raise ValueError("TiedTranspose does not support layers with bias.")
        return F.linear(x, self.linear.weight.t(), None)

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight.t()

    @property
    def bias(self) -> torch.Tensor:
        return self.linear.bias


class SparseAutoEncoder(Module):
    """
    This module implements the Sparse AutoEncoder architecture from
    `Beyond Matryoshka: Revisiting Sparse Coding for Adaptive Representation
    <https://huggingface.co/papers/2503.01776>`_, with an optional SAE-SPLADE
    forward path from `From Tokens to Concepts: Leveraging SAE for SPLADE
    <https://huggingface.co/papers/2604.21511>`_.

    The encoder math is the same in both modes (pre-bias, tied decoder, top-K, AuxK,
    dead-neuron stats); only the I/O contract differs:

    * ``mode="csr"`` (default) — the CSR pipeline. Reads a per-sentence vector from
      ``features["sentence_embedding"]`` of shape ``(batch, input_dim)`` and writes a
      top-K sparse vector back to the same key (shape ``(batch, hidden_dim)``).
    * ``mode="splade"`` — the SAE-SPLADE pipeline. Reads per-token hidden states from
      ``features["token_embeddings"]`` of shape ``(batch, seq_length, input_dim)`` and
      writes top-K SAE latents back to the same key (shape
      ``(batch, seq_length, hidden_dim)``), ready for a downstream
      :class:`SpladePooling`. In this mode ``k=0`` (or any value ``>= hidden_dim``)
      skips the top-K mask entirely — sparsity is then produced by the downstream
      ReLU + log1p + max-pool, as in the SPLADE fine-tuning phase of the paper.

    Training-mode forward exposes the usual reconstruction / auxiliary intermediates
    under prefixes that match the mode (``sentence_embedding_*`` /
    ``decoded_embedding_*`` for csr, ``token_embeddings_*`` /
    ``decoded_token_embeddings_*`` for splade).

    Args:
        input_dim: Dimension of the input embeddings.
        hidden_dim: Dimension of the hidden layers. Defaults to 512.
        k: Number of top values to keep in the final sparse representation. Defaults to 8.
        k_aux: Number of top values to keep for auxiliary loss calculation. Defaults to 512.
        normalize: Whether to apply layer normalization to the input embeddings. Defaults to False.
        dead_threshold: Threshold for dead neurons. Neurons with non-zero activations below this threshold are considered dead. Defaults to 30.
        mode: Which forward path to use — ``"csr"`` (default, sentence-level CSR pipeline)
            or ``"splade"`` (token-level SAE-SPLADE pipeline). The choice is persisted in
            the config so saved models round-trip correctly.
    """

    config_keys = ["input_dim", "hidden_dim", "k", "k_aux", "normalize", "dead_threshold", "mode"]

    forward_kwargs = {"max_active_dims"}

    # Feature dict key conventions per mode. ``input`` is the key the SAE reads from and
    # writes its top-K output back to; the rest are the training-mode intermediates.
    _FEATURE_KEYS: dict[str, dict[str, str]] = {
        "csr": {
            "input": "sentence_embedding",
            "backbone": "sentence_embedding_backbone",
            "encoded": "sentence_embedding_encoded",
            "encoded_4k": "sentence_embedding_encoded_4k",
            "auxiliary": "auxiliary_embedding",
            "decoded_k": "decoded_embedding_k",
            "decoded_4k": "decoded_embedding_4k",
            "decoded_aux": "decoded_embedding_aux",
            "decoded_k_pre_bias": "decoded_embedding_k_pre_bias",
        },
        "splade": {
            "input": "token_embeddings",
            "backbone": "token_embeddings_backbone",
            "encoded": "token_embeddings_encoded",
            "encoded_4k": "token_embeddings_encoded_4k",
            "auxiliary": "auxiliary_token_embeddings",
            "decoded_k": "decoded_token_embeddings_k",
            "decoded_4k": "decoded_token_embeddings_4k",
            "decoded_aux": "decoded_token_embeddings_aux",
            "decoded_k_pre_bias": "decoded_token_embeddings_k_pre_bias",
        },
    }

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        k: int = 8,
        k_aux: int = 512,
        normalize: bool = False,
        dead_threshold: int = 30,
        mode: SparseAutoEncoderMode = "csr",
    ) -> None:
        super().__init__()
        if mode not in self._FEATURE_KEYS:
            raise ValueError(f"mode must be one of {sorted(self._FEATURE_KEYS)}, got {mode!r}")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dead_threshold = dead_threshold
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))
        self.encoder: nn.Module = nn.Linear(input_dim, hidden_dim, bias=False)
        self.latent_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.decoder: TiedTranspose = TiedTranspose(self.encoder)
        self.k = k
        self.k_aux = k_aux
        self.normalize = normalize
        self.mode = mode

        self.stats_last_nonzero: torch.Tensor
        self.register_buffer("stats_last_nonzero", torch.zeros(hidden_dim, dtype=torch.long))

        def auxk_mask_fn(x):
            dead_mask = self.stats_last_nonzero > dead_threshold
            x.data *= dead_mask  # inplace to save memory
            return x

        self.auxk_mask_fn = auxk_mask_fn

    def encode_pre_act(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input data (shape: [batch, input_dim])
        :param latent_slice: slice of latents to compute
            Example: latent_slice = slice(0, 10) to compute only the first 10 latents.
        :return: autoencoder latents before activation (shape: [batch, hidden_dim])
        """
        x = x - self.pre_bias
        latents_pre_act = F.linear(x, self.encoder.weight, self.latent_bias)
        return latents_pre_act

    def LN(self, x: torch.Tensor, eps: float = 1e-5):
        mu = x.mean(dim=-1, keepdim=True)
        x = x - mu
        std = x.std(dim=-1, keepdim=True)
        x = x / (std + eps)
        return x, mu, std

    def prepare(self, x: torch.Tensor):
        if not self.normalize:
            return x, dict()
        x, mu, std = self.LN(x)
        return x, dict(mu=mu, std=std)

    def top_k(
        self, x: torch.Tensor, k: int | None = None, compute_aux: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        :param x: input data (shape: [batch, input_dim])
        :return: tuple of (top-k latents (shape: [batch, hidden_dim]), auxiliary latents or None)
        """
        if k is None:
            k = self.k
        topk = torch.topk(x, k=k, dim=-1)
        z_topk = torch.zeros_like(x)
        z_topk.scatter_(-1, topk.indices, topk.values)
        latents_k = F.relu(z_topk)
        ## set num nonzero stat ##
        tmp = torch.zeros_like(self.stats_last_nonzero)
        tmp.scatter_add_(
            0,
            topk.indices.reshape(-1),
            (topk.values > 1e-5).to(tmp.dtype).reshape(-1),
        )
        self.stats_last_nonzero *= 1 - tmp.clamp(max=1)
        self.stats_last_nonzero += 1
        ## end stats ##

        latents_auxk = None
        if self.k_aux and compute_aux:
            aux_topk = torch.topk(
                input=self.auxk_mask_fn(x),
                k=self.k_aux,
            )
            z_auxk = torch.zeros_like(x)
            z_auxk.scatter_(-1, aux_topk.indices, aux_topk.values)
            latents_auxk = F.relu(z_auxk)
        return latents_k, latents_auxk

    def decode(self, latents: torch.Tensor, info=None) -> torch.Tensor:
        """
        :param latents: autoencoder latents (shape: [batch, hidden_dim])
        :return: reconstructed data (shape: [batch, n_inputs])
        """

        ret = self.decoder(latents) + self.pre_bias

        if self.normalize:
            assert info is not None
            ret = ret * info["std"] + info["mu"]
        return ret

    def _topk_disabled(self, k: int) -> bool:
        # ``mode="splade"`` only: a non-positive ``k`` (or one as wide as the dictionary)
        # means "no top-K mask" — the downstream SpladePooling enforces sparsity instead.
        return k <= 0 or k >= self.hidden_dim

    def forward(
        self, features: dict[str, torch.Tensor], max_active_dims: int | None = None
    ) -> dict[str, torch.Tensor]:
        k = max_active_dims if max_active_dims is not None else self.k
        keys = self._FEATURE_KEYS[self.mode]
        x = features[keys["input"]]

        x, info = self.prepare(x)
        latents_pre_act = self.encode_pre_act(x)

        # In splade mode the user may skip the top-K mask entirely; csr mode always uses
        # top-K because top-K is the only thing producing sparsity in that pipeline.
        skip_topk = self.mode == "splade" and self._topk_disabled(k)

        # If the model is in inference mode, we don't need to e.g. compute the 4k, auxk, or apply the decoder
        if torch.is_inference_mode_enabled():
            latents_k = torch.relu(latents_pre_act) if skip_topk else self.top_k(latents_pre_act, k, compute_aux=False)[0]
            features[keys["input"]] = latents_k
            return features

        if skip_topk:
            latents_k = torch.relu(latents_pre_act)
            latents_4k = latents_k
            latents_auxk = None
        else:
            latents_k, latents_auxk = self.top_k(latents_pre_act, k)
            latents_4k, _ = self.top_k(latents_pre_act, min(4 * k, self.hidden_dim))

        recons_k = self.decode(latents_k, info)
        recons_4k = self.decode(latents_4k, info)
        recons_aux = self.decode(latents_auxk, info) if latents_auxk is not None else None

        # Update the features dictionary
        features.update(
            {
                keys["backbone"]: x,
                keys["encoded"]: latents_pre_act,
                keys["encoded_4k"]: latents_4k,
                keys["auxiliary"]: latents_auxk,
                keys["decoded_k"]: recons_k,
                keys["decoded_4k"]: recons_4k,
                keys["decoded_aux"]: recons_aux,
                keys["decoded_k_pre_bias"]: recons_k - self.pre_bias,
            }
        )
        features[keys["input"]] = latents_k
        return features

    def save(self, output_path, safe_serialization: bool = True) -> None:
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

    def __repr__(self):
        return f"SparseAutoEncoder({self.get_config_dict()})"

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding. Warning: the number of non-zero elements in the embedding is only k out of the hidden_dim.

        Returns:
            int: Dimension of the embedding
        """
        return self.hidden_dim
