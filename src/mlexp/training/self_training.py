from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from mlexp.adapters.factory import wrap_model

logger = logging.getLogger("mlexp.self_training")


@dataclass
class SelfTrainingResult:
    """Diagnostics collected across self-training iterations."""

    n_iterations: int
    pseudo_labels_added_per_iter: list[int] = field(default_factory=list)
    confidence_stats_per_iter: list[dict[str, float]] = field(default_factory=list)
    validation_scores: list[float] = field(default_factory=list)


class SelfTrainer:
    """
    Model-agnostic self-training loop.

    Algorithm
    ---------
    1. Train model on (X_labeled, y_labeled) with optional sample_weight.
    2. Predict on X_unlabeled; keep predictions with confidence ≥ threshold.
    3. Add accepted pseudo-labeled samples to training set.
    4. Repeat for max_iter rounds or until no new samples are accepted.

    Validation is performed each round on X_val / y_val (clean labels only).
    Pseudo-labels are NEVER added to the validation set.

    Notes
    -----
    - For classifiers the model must support predict_proba.
    - Confidence = max class probability.
    - sample_weight for pseudo-labeled samples is optionally down-weighted
      by pseudo_weight_factor (default 1.0 = equal weight).
    """

    def __init__(
        self,
        model_fn: Callable[..., Any],
        model_params: dict[str, Any],
        confidence_threshold: float = 0.9,
        max_iter: int = 10,
        pseudo_weight_factor: float = 1.0,
        score_fn: Callable | None = None,
    ) -> None:
        self.model_fn = model_fn
        self.model_params = model_params
        self.confidence_threshold = confidence_threshold
        self.max_iter = max_iter
        self.pseudo_weight_factor = pseudo_weight_factor
        self.score_fn = score_fn

    def fit(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> tuple[Any, SelfTrainingResult]:
        """
        Run self-training and return (final_model_adapter, result_diagnostics).
        """
        X_lab = np.asarray(X_labeled, dtype=float)
        y_lab = np.asarray(y_labeled)
        X_unlab = np.asarray(X_unlabeled, dtype=float)
        sw = np.asarray(sample_weight) if sample_weight is not None else None

        result = SelfTrainingResult(n_iterations=0)
        remaining_idx = np.arange(len(X_unlab))

        for iteration in range(self.max_iter):
            adapter = wrap_model(self.model_fn(**self.model_params))
            adapter.fit(X_lab, y_lab, sample_weight=sw)

            if len(remaining_idx) == 0:
                logger.info("iter %d: no unlabeled data left, stopping.", iteration)
                break

            X_remaining = X_unlab[remaining_idx]
            try:
                proba = adapter.predict_proba(X_remaining)
            except NotImplementedError:
                logger.warning(
                    "Model does not support predict_proba; self-training requires it."
                )
                break

            confidence = proba.max(axis=1)
            pseudo_labels = proba.argmax(axis=1)

            accept_mask = confidence >= self.confidence_threshold
            n_accepted = int(accept_mask.sum())

            # Diagnostics
            result.pseudo_labels_added_per_iter.append(n_accepted)
            result.confidence_stats_per_iter.append(
                {
                    "mean": float(confidence.mean()),
                    "min": float(confidence.min()),
                    "max": float(confidence.max()),
                    "pct_accepted": float(accept_mask.mean()),
                }
            )

            if X_val is not None and y_val is not None and self.score_fn is not None:
                val_score = self.score_fn(adapter, X_val, y_val)
                result.validation_scores.append(float(val_score))
                logger.info(
                    "iter %d: accepted=%d  val_score=%.4f",
                    iteration,
                    n_accepted,
                    val_score,
                )
            else:
                logger.info("iter %d: accepted=%d", iteration, n_accepted)

            if n_accepted == 0:
                logger.info("iter %d: no confident pseudo-labels, stopping.", iteration)
                break

            # Add accepted pseudo-labeled samples to training set
            accepted_global_idx = remaining_idx[accept_mask]
            X_pseudo = X_unlab[accepted_global_idx]
            y_pseudo = pseudo_labels[accept_mask]

            X_lab = np.vstack([X_lab, X_pseudo])
            y_lab = np.concatenate([y_lab, y_pseudo])

            if sw is not None:
                pseudo_sw = np.full(n_accepted, self.pseudo_weight_factor)
                sw = np.concatenate([sw, pseudo_sw])

            remaining_idx = remaining_idx[~accept_mask]
            result.n_iterations = iteration + 1

        # Final model trained on all accumulated data
        final_adapter = wrap_model(self.model_fn(**self.model_params))
        final_adapter.fit(X_lab, y_lab, sample_weight=sw)

        return final_adapter, result
