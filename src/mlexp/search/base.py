from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from sklearn.model_selection import BaseCrossValidator


@dataclass
class SearchResult:
    best_params: dict[str, Any]
    best_score: float
    best_score_std: float
    all_results: list[dict[str, Any]] = field(default_factory=list)


class BaseSearch(ABC):
    """
    Contract: takes a model factory, param space, data, CV splitter and a
    scoring callable → returns a SearchResult.

    The scoring callable must be:   score_fn(model_adapter, X, y) -> float
    Higher is always better (callers must negate if using a loss).
    """

    @abstractmethod
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
        ...
