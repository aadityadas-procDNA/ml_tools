from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.model_selection import BaseCrossValidator

from mlexp.evaluation.base import BaseEvaluator, EvaluationResult
from mlexp.search.base import BaseSearch


class CVEvaluator(BaseEvaluator):
    """
    Simple cross-validated evaluation.

    Runs inner search once on the full dataset.  Best params are used to
    train on each fold and evaluate.  Note: this shares data between tuning
    and evaluation → mild optimism bias, but cheap and practical.
    """

    def __init__(self, cv: BaseCrossValidator) -> None:
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
        search_result = searcher.search(
            model_fn, param_space, X, y, self.cv, score_fn, fit_params
        )

        from mlexp.adapters.factory import wrap_model
        from mlexp.search.grid_search import _cv_fold_metrics

        model = wrap_model(model_fn(**search_result.best_params))
        fold_scores, extra = _cv_fold_metrics(
            model, X, y, self.cv, score_fn, additional_metrics, fit_params
        )
        outer_scores = fold_scores.tolist()

        return EvaluationResult(
            outer_scores=outer_scores,
            mean_score=float(np.mean(outer_scores)),
            std_score=float(np.std(outer_scores)),
            best_params_per_fold=[search_result.best_params],
            inner_results=[search_result],
            additional_metric_scores=extra,
        )
