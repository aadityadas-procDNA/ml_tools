from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.model_selection import BaseCrossValidator

from mlexp.adapters.factory import wrap_model
from mlexp.evaluation.base import BaseEvaluator, EvaluationResult
from mlexp.search.base import BaseSearch


class HoldoutEvaluator(BaseEvaluator):
    """
    Fixed train/validation/holdout split evaluation.

    Workflow:
      1. Tune hyperparameters via inner CV on X_train / y_train
      2. Re-train best model on full X_train
      3. Evaluate once on X_holdout (the outer test set)

    X and y passed to evaluate() are the training portion.
    X_holdout / y_holdout are passed at construction time.
    """

    def __init__(
        self,
        X_holdout: np.ndarray,
        y_holdout: np.ndarray,
        cv: BaseCrossValidator,
    ) -> None:
        self.X_holdout = X_holdout
        self.y_holdout = y_holdout
        self.cv = cv

    def evaluate(
        self,
        model_fn: Callable[..., Any],
        param_space: dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        searcher: BaseSearch,
        score_fn: Callable,
        fit_params: dict[str, Any] | None = None,
        additional_metrics: dict[str, Callable] | None = None,
    ) -> EvaluationResult:
        # score_fn drives selection — additional_metrics are NOT passed to the searcher
        result = searcher.search(
            model_fn, param_space, X, y, self.cv, score_fn, fit_params
        )

        model = wrap_model(model_fn(**result.best_params))
        sw = fit_params.get("sample_weight") if fit_params else None
        model.fit(X, y, sample_weight=sw)

        holdout_score = float(score_fn(model, self.X_holdout, self.y_holdout))

        extra: dict[str, list[float]] = {}
        for name, fn in (additional_metrics or {}).items():
            extra[name] = [float(fn(model, self.X_holdout, self.y_holdout))]

        return EvaluationResult(
            outer_scores=[holdout_score],
            mean_score=holdout_score,
            std_score=0.0,
            best_params_per_fold=[result.best_params],
            inner_results=[result],
            additional_metric_scores=extra,
        )
