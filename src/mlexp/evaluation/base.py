from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from sklearn.model_selection import BaseCrossValidator

from mlexp.search.base import BaseSearch


@dataclass
class EvaluationResult:
    """
    Aggregated result from an evaluation strategy.

    outer_scores             : per-fold (or single) primary scores on held-out test data
    mean_score               : mean of outer_scores
    std_score                : std of outer_scores
    best_params_per_fold     : params chosen by the inner search per outer fold
    inner_results            : raw SearchResult objects per outer fold
    additional_metric_scores : {metric_name: [fold_score, ...]} for reporting-only metrics
    """

    outer_scores: list[float]
    mean_score: float
    std_score: float
    best_params_per_fold: list[dict[str, Any]] = field(default_factory=list)
    inner_results: list[Any] = field(default_factory=list)
    additional_metric_scores: dict[str, list[float]] = field(default_factory=dict)


class BaseEvaluator(ABC):
    @abstractmethod
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
        ...
