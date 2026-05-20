from __future__ import annotations

import logging
from enum import Enum

import torch.nn as nn
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState

from sentence_transformers.sparse_encoder.losses.splade import SpladeLoss
from sentence_transformers.sparse_encoder.modules.sparse_auto_encoder import SparseAutoEncoder
from sentence_transformers.sparse_encoder.training_args import SparseEncoderTrainingArguments

logger = logging.getLogger(__name__)


class SpladeDecoderNormalizationCallback(TrainerCallback):
    """Renormalises every splade-mode :class:`SparseAutoEncoder`'s decoder weights
    to unit row-norm after each optimizer step.

    Pairs with the parallel-gradient-stripping hook on ``W_dec`` so the decoder
    rows stay on the unit sphere throughout training — the constraint that makes
    each learned latent a well-defined "concept direction" in hidden space.

    Required for ``SparseAutoEncoder(mode="splade")`` modules; csr-mode SAEs are
    safely no-op'd by the underlying :meth:`SparseAutoEncoder.normalize_decoder_`.

    Args:
        model: The model whose modules should be scanned for splade-mode SAEs.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self._splade_saes = [
            module for module in model.modules() if isinstance(module, SparseAutoEncoder) and module.mode == "splade"
        ]
        if not self._splade_saes:
            logger.warning(
                "SpladeDecoderNormalizationCallback registered, but the model contains no "
                "SparseAutoEncoder(mode='splade') modules. The callback will have no effect."
            )

    def on_step_end(
        self,
        args: SparseEncoderTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        for sae in self._splade_saes:
            sae.normalize_decoder_()


class SchedulerType(Enum):
    """Types of schedulers for weight parameters in SpladeLoss"""

    LINEAR = "linear"
    QUADRATIC = "quadratic"


class SpladeRegularizerWeightSchedulerCallback(TrainerCallback):
    def __init__(
        self,
        loss: SpladeLoss,
        scheduler_type: str | SchedulerType = SchedulerType.QUADRATIC,
        warmup_ratio: float = 1 / 3,
    ):
        """
        Callback that updates the query_regularizer_weight and document_regularizer_weight parameters of SpladeLoss
        based on a schedule.

        The scheduler gradually increases the weight values from 0 to their max value
        within the specified warmup ratio of the total training steps.

        Args:
            loss (SpladeLoss): SpladeLoss instance to be updated
            scheduler_type (str): Type of scheduler ('linear' or 'quadratic')
            warmup_ratio (float): Ratio of total steps to reach max weight values (default: 1/3)
        """
        super().__init__()

        if isinstance(scheduler_type, str):
            try:
                scheduler_type = SchedulerType(scheduler_type.lower())
            except ValueError:
                logger.warning(
                    f"Invalid scheduler_type: {scheduler_type}. Using default: {SchedulerType.QUADRATIC.value}"
                )
                scheduler_type = SchedulerType.QUADRATIC

        self.scheduler_type = scheduler_type

        # Validate warmup_ratio is between 0 and 1
        if not 0 < warmup_ratio <= 1:
            logger.warning(f"warmup_ratio should be between 0 and 1, got {warmup_ratio}. Setting to default 1/3.")
            warmup_ratio = 1 / 3

        # Validate loss is an instance of SpladeLoss
        if not isinstance(loss, SpladeLoss):
            logger.warning(
                f"SpladeRegularizerWeightSchedulerCallback is only compatible with SpladeLoss, "
                f"but got {type(loss).__name__}. This callback won't have any effect."
            )
            raise ValueError("loss must be an instance of SpladeLoss")
        self.loss = loss
        self.max_document_regularizer_weight = self.loss.document_regularizer_weight
        self.max_query_regularizer_weight = self.loss.query_regularizer_weight
        self.warmup_ratio = warmup_ratio
        self._current_query_regularizer_weight = 0.0 if self.max_query_regularizer_weight is not None else None
        self._current_document_regularizer_weight = 0.0
        self.total_steps = None
        self.warmup_steps = None

    def on_train_begin(
        self,
        args: SparseEncoderTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Initialize the scheduler at the beginning of training."""
        # Calculate total steps and warmup steps
        if hasattr(state, "max_steps") and state.max_steps > 0:
            self.total_steps = state.max_steps
        elif hasattr(state, "num_train_epochs") and hasattr(state, "num_update_steps_per_epoch"):
            self.total_steps = state.num_update_steps_per_epoch * state.num_train_epochs
        else:
            logger.warning("Cannot determine total steps from TrainerState. Weight scheduling may not work properly.")
            return

        self.warmup_steps = int(self.total_steps * self.warmup_ratio)
        if self.warmup_steps <= 0:
            self.warmup_steps = 1  # Ensure at least one step for warmup

        # Set initial weight values
        self.loss.query_regularizer_weight = self._current_query_regularizer_weight
        self.loss.document_regularizer_weight = self._current_document_regularizer_weight

    def _calculate_weight_value(self, step: int, max_value: float) -> float:
        """Calculate the weight value based on the current step and scheduler type."""
        if self.warmup_steps is None or step >= self.warmup_steps or max_value is None:
            return max_value

        ratio = step / max(self.warmup_steps, 1)  # Avoid division by zero

        if self.scheduler_type == SchedulerType.LINEAR:
            return max_value * ratio
        elif self.scheduler_type == SchedulerType.QUADRATIC:
            return max_value * (ratio**2)
        else:
            logger.warning(f"Unknown scheduler type: {self.scheduler_type}. Using quadratic.")
            return max_value * (ratio**2)

    def on_step_begin(
        self,
        args: SparseEncoderTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Update weight values at the beginning of each step."""
        if self.total_steps is None or self.warmup_steps is None:
            return

        # Get current step
        step = state.global_step

        # Calculate new weight values
        new_query_regularizer_weight = self._calculate_weight_value(step, self.max_query_regularizer_weight)
        new_document_regularizer_weight = self._calculate_weight_value(step, self.max_document_regularizer_weight)

        # Update weight values only if they've changed
        if (
            new_query_regularizer_weight != self._current_query_regularizer_weight
            or new_document_regularizer_weight != self._current_document_regularizer_weight
        ):
            self.loss.query_regularizer_weight = new_query_regularizer_weight
            self.loss.document_regularizer_weight = new_document_regularizer_weight

            # Store current values
            self._current_query_regularizer_weight = new_query_regularizer_weight
            self._current_document_regularizer_weight = new_document_regularizer_weight

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Log the current weight values."""
        logs["document_regularizer_weight"] = self._current_document_regularizer_weight
        if self._current_query_regularizer_weight is not None:
            logs["query_regularizer_weight"] = self._current_query_regularizer_weight
