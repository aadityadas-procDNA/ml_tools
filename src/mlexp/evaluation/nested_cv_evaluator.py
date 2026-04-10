from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.model_selection import BaseCrossValidator

from mlexp.adapters.factory import wrap_model
from mlexp.evaluation.base import BaseEvaluator, EvaluationResult
from mlexp.search.base import BaseSearch


class NestedCVEvaluator(BaseEvaluator):
    """
    Nested cross-validation.

    Outer loop  → unbiased performance estimation
    Inner loop  → hyperparameter tuning (via pluggable searcher)

    Each outer fold:
      1. Split data into train_outer / test_outer
      2. Run inner search on train_outer only → best_params
      3. Retrain on full train_outer with best_params
      4. Evaluate on test_outer

    Final performance = mean ± std of outer fold scores.

    Note: this does NOT produce a single deployable model.
    Use the returned best_params_per_fold to derive a final config
    (e.g. mode or median of params) and retrain on all data.
    """

    def __init__(
        self,
        outer_cv: BaseCrossValidator,
        inner_cv: BaseCrossValidator,
    ) -> None:
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv

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
        outer_scores: list[float] = []
        best_params_per_fold: list[dict] = []
        inner_results = []
        extra: dict[str, list[float]] = {k: [] for k in (additional_metrics or {})}

        for train_idx, test_idx in self.outer_cv.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            # Slice fit_params (e.g. sample_weight) to outer train indices
            inner_fit_params = _slice_fit_params(fit_params, train_idx)

            # Inner search on outer training data only — score_fn drives selection
            result = searcher.search(
                model_fn,
                param_space,
                X_tr,
                y_tr,
                self.inner_cv,
                score_fn,
                inner_fit_params,
            )
            inner_results.append(result)
            best_params_per_fold.append(result.best_params)

            # Retrain with best params on full outer train set
            model = wrap_model(model_fn(**result.best_params))
            sw_tr = inner_fit_params.get("sample_weight") if inner_fit_params else None
            model.fit(X_tr, y_tr, sample_weight=sw_tr)

            outer_scores.append(float(score_fn(model, X_te, y_te)))

            # Additional metrics on outer test fold — reporting only
            for name, fn in (additional_metrics or {}).items():
                extra[name].append(float(fn(model, X_te, y_te)))

        return EvaluationResult(
            outer_scores=outer_scores,
            mean_score=float(np.mean(outer_scores)),
            std_score=float(np.std(outer_scores)),
            best_params_per_fold=best_params_per_fold,
            inner_results=inner_results,
            additional_metric_scores=extra,
        )


def _slice_fit_params(
    fit_params: dict | None, idx: np.ndarray
) -> dict | None:
    if not fit_params:
        return fit_params
    sliced: dict = {}
    for k, v in fit_params.items():
        if hasattr(v, "__len__") and hasattr(v, "__getitem__"):
            sliced[k] = v[idx]
        else:
            sliced[k] = v
    return sliced
