from __future__ import annotations

import random
from typing import Any, Callable

import numpy as np
from sklearn.model_selection import BaseCrossValidator

from mlexp.adapters.factory import wrap_model
from mlexp.search.base import BaseSearch, SearchResult
from mlexp.search.grid_search import _cv_scores


class RandomSearch(BaseSearch):
    """
    Random search over a parameter space.

    param_space values can be:
      - a list           → sampled uniformly
      - a scipy rv_frozen  → uses .rvs()
      - a callable       → called with no args
    """

    def __init__(self, n_iter: int = 20, random_state: int | None = None) -> None:
        self.n_iter = n_iter
        self.random_state = random_state

    def search(
        self,
        model_fn: Callable[..., Any],
        param_space: dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        cv: BaseCrossValidator,
        score_fn: Callable,
        fit_params: dict[str, Any] | None = None,
    ) -> SearchResult:
        rng = random.Random(self.random_state)

        best_score = -np.inf
        best_params: dict = {}
        best_std: float = 0.0
        all_results: list[dict] = []

        for _ in range(self.n_iter):
            params = {k: _sample(v, rng) for k, v in param_space.items()}
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


def _sample(dist, rng: random.Random):
    if isinstance(dist, list):
        return rng.choice(dist)
    if callable(getattr(dist, "rvs", None)):
        return dist.rvs()
    if callable(dist):
        return dist()
    return dist
