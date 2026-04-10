from __future__ import annotations

from itertools import product
from typing import Any, Callable

import numpy as np
from sklearn.model_selection import BaseCrossValidator, cross_val_score

from mlexp.adapters.factory import wrap_model
from mlexp.search.base import BaseSearch, SearchResult


class GridSearch(BaseSearch):
    """Exhaustive grid search over a discrete parameter grid."""

    def search(
        self,
        model_fn: Callable[..., Any],
        param_space: dict[str, list],
        X: np.ndarray,
        y: np.ndarray,
        cv: BaseCrossValidator,
        score_fn: Callable,
        fit_params: dict[str, Any] | None = None,
    ) -> SearchResult:
        keys = list(param_space.keys())
        values = list(param_space.values())

        best_score = -np.inf
        best_params: dict = {}
        best_std: float = 0.0
        all_results: list[dict] = []

        for combo in product(*values):
            params = dict(zip(keys, combo))
            model = wrap_model(model_fn(**params))

            scores = _cv_scores(model, X, y, cv, score_fn, fit_params)
            mean, std = float(np.mean(scores)), float(np.std(scores))

            all_results.append({"params": params, "mean": mean, "std": std})

            if mean > best_score:
                best_score = mean
                best_std = std
                best_params = params

        return SearchResult(
            best_params=best_params,
            best_score=best_score,
            best_score_std=best_std,
            all_results=all_results,
        )


def _cv_scores(
    adapter,
    X: np.ndarray,
    y: np.ndarray,
    cv: BaseCrossValidator,
    score_fn: Callable,
    fit_params: dict | None,
) -> np.ndarray:
    """Run CV folds manually so we can pass sample_weight through score_fn."""
    fold_scores, _ = _cv_fold_metrics(adapter, X, y, cv, score_fn, None, fit_params)
    return fold_scores


def _cv_fold_metrics(
    adapter,
    X: np.ndarray,
    y: np.ndarray,
    cv: BaseCrossValidator,
    score_fn: Callable,
    additional_metrics: dict[str, Callable] | None,
    fit_params: dict | None,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    """
    Run CV folds and return both the primary scores and any additional metric scores.

    Returns
    -------
    primary_scores : np.ndarray of shape (n_folds,)
    extra_scores   : {metric_name: [fold_score, ...]}  — empty dict if no additional_metrics
    """
    fold_scores: list[float] = []
    extra_scores: dict[str, list[float]] = {k: [] for k in (additional_metrics or {})}

    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        sw = fit_params.get("sample_weight") if fit_params else None
        sw_tr = sw[train_idx] if sw is not None else None

        adapter.fit(X_tr, y_tr, sample_weight=sw_tr)
        fold_scores.append(float(score_fn(adapter, X_val, y_val)))

        for name, fn in (additional_metrics or {}).items():
            extra_scores[name].append(float(fn(adapter, X_val, y_val)))

    return np.array(fold_scores), extra_scores
