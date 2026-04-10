from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ExperimentConfig:
    """
    Central configuration object for a single experiment run.

    model_fn          : Callable(**params) -> sklearn-compatible model
    model_name        : Human-readable name for tracking
    param_space       : Hyperparameter search space (format depends on searcher)
    search_strategy   : "grid" | "random" | "optuna"
    search_kwargs     : Extra kwargs forwarded to get_searcher()
                        e.g. {"n_iter": 30} for random, {"n_trials": 50} for optuna
    evaluation_strategy: "cv" | "nested_cv" | "holdout"
    evaluation_kwargs : Extra kwargs forwarded to get_evaluator()
                        e.g. {"cv": KFold(5)} for cv/holdout
                             {"outer_cv": ..., "inner_cv": ...} for nested_cv
                             {"X_holdout": ..., "y_holdout": ...} for holdout
    score_fn          : Callable(model_adapter, X, y) -> float (higher = better)
                        This is the ONLY metric used for hyperparameter selection.
    additional_metrics: Optional dict of extra metrics for reporting only.
                        These are computed on outer test folds but NEVER used for
                        selection.  Format: {"name": Callable(adapter, X, y) -> float}
                        e.g. {"roc_auc": lambda a, X, y: roc_auc_score(y, a.predict_proba(X)[:,1]),
                              "f1":      lambda a, X, y: f1_score(y, a.predict(X))}
    fit_params        : Optional dict passed to model.fit()
                        e.g. {"sample_weight": weights_array}
    tags              : Free-form metadata stored alongside results
    run_id            : If None, auto-generated from model_name + timestamp
    """

    model_fn: Callable[..., Any]
    model_name: str
    param_space: dict[str, Any]
    score_fn: Callable
    additional_metrics: dict[str, Callable] | None = None
    search_strategy: str = "grid"
    search_kwargs: dict[str, Any] = field(default_factory=dict)
    evaluation_strategy: str = "cv"
    evaluation_kwargs: dict[str, Any] = field(default_factory=dict)
    fit_params: dict[str, Any] | None = None
    tags: dict[str, str] = field(default_factory=dict)
    run_id: str | None = None
