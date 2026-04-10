from __future__ import annotations

from sklearn.model_selection import BaseCrossValidator

from mlexp.evaluation.base import BaseEvaluator
from mlexp.evaluation.cv_evaluator import CVEvaluator
from mlexp.evaluation.holdout_evaluator import HoldoutEvaluator
from mlexp.evaluation.nested_cv_evaluator import NestedCVEvaluator


def get_evaluator(strategy: str, **kwargs) -> BaseEvaluator:
    """
    Factory for evaluation strategies.

    strategy: "cv" | "nested_cv" | "holdout"

    kwargs for "cv":
        cv: BaseCrossValidator

    kwargs for "nested_cv":
        outer_cv: BaseCrossValidator
        inner_cv: BaseCrossValidator

    kwargs for "holdout":
        X_holdout, y_holdout: arrays
        cv: BaseCrossValidator  (used for inner tuning)
    """
    key = strategy.lower()
    if key == "cv":
        return CVEvaluator(cv=kwargs["cv"])
    if key == "nested_cv":
        return NestedCVEvaluator(
            outer_cv=kwargs["outer_cv"],
            inner_cv=kwargs["inner_cv"],
        )
    if key == "holdout":
        return HoldoutEvaluator(
            X_holdout=kwargs["X_holdout"],
            y_holdout=kwargs["y_holdout"],
            cv=kwargs["cv"],
        )
    raise ValueError(
        f"Unknown evaluation strategy '{strategy}'. Choose from: cv, nested_cv, holdout"
    )
