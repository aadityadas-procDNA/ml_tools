from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from mlexp.config import ExperimentConfig
from mlexp.evaluation.base import EvaluationResult
from mlexp.evaluation.factory import get_evaluator
from mlexp.search.factory import get_searcher
from mlexp.tracking.logger import ExperimentLogger, ExperimentRecord


class Experiment:
    """
    Top-level orchestrator.

    Usage
    -----
    from sklearn.model_selection import KFold, StratifiedKFold
    from xgboost import XGBClassifier
    from mlexp import Experiment, ExperimentConfig
    from mlexp.weighting import compute_distance_weights

    weights = compute_distance_weights(X_train, X_reference)

    config = ExperimentConfig(
        model_fn=lambda **p: XGBClassifier(**p),
        model_name="xgboost",
        param_space={
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1],
            "n_estimators": [100, 300],
        },
        score_fn=lambda adapter, X, y: accuracy_score(y, adapter.predict(X)),
        search_strategy="grid",
        evaluation_strategy="nested_cv",
        evaluation_kwargs={
            "outer_cv": StratifiedKFold(5, shuffle=True, random_state=42),
            "inner_cv": StratifiedKFold(3, shuffle=True, random_state=42),
        },
        fit_params={"sample_weight": weights},
        tags={"dataset": "customer_churn", "experiment": "distance_weighted"},
    )

    exp = Experiment(log_dir="runs/")
    result, record = exp.run(config, X_train, y_train)
    print(f"Score: {record.mean_score:.4f} ± {record.std_score:.4f}")
    """

    def __init__(self, log_dir: str | Path = "mlexp_runs") -> None:
        self.logger = ExperimentLogger(log_dir)

    def run(
        self,
        config: ExperimentConfig,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[EvaluationResult, ExperimentRecord]:
        run_id = config.run_id or f"{config.model_name}_{int(time.time())}_{uuid.uuid4().hex[:6]}"

        searcher = get_searcher(config.search_strategy, **config.search_kwargs)
        evaluator = get_evaluator(config.evaluation_strategy, **config.evaluation_kwargs)

        result: EvaluationResult = evaluator.evaluate(
            model_fn=config.model_fn,
            param_space=config.param_space,
            X=np.asarray(X),
            y=np.asarray(y),
            searcher=searcher,
            score_fn=config.score_fn,
            fit_params=config.fit_params,
            additional_metrics=config.additional_metrics,
        )

        # Derive a single "consensus" param set from all outer folds
        consensus_params = _consensus_params(result.best_params_per_fold)

        # Summarise additional metrics: mean ± std across folds
        additional_metrics_summary = {
            name: {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "scores": scores,
            }
            for name, scores in result.additional_metric_scores.items()
        }

        record = ExperimentRecord(
            run_id=run_id,
            model_name=config.model_name,
            search_strategy=config.search_strategy,
            evaluation_strategy=config.evaluation_strategy,
            best_params=consensus_params,
            mean_score=result.mean_score,
            std_score=result.std_score,
            outer_scores=result.outer_scores,
            best_params_per_fold=result.best_params_per_fold,
            additional_metrics=additional_metrics_summary,
            tags=config.tags,
        )

        self.logger.log(record)
        return result, record


def _consensus_params(params_list: list[dict]) -> dict:
    """
    Derive a single param dict from a list of per-fold best params.

    Numeric: median across folds
    Categorical/string: mode (most frequent value)
    """
    if not params_list:
        return {}
    if len(params_list) == 1:
        return params_list[0]

    keys = params_list[0].keys()
    consensus: dict[str, Any] = {}

    for k in keys:
        values = [p[k] for p in params_list if k in p]
        if not values:
            continue
        if isinstance(values[0], (int, float)):
            median_val = float(np.median(values))
            # Snap back to int if the original type was int
            if isinstance(values[0], int):
                median_val = int(round(median_val))
            consensus[k] = median_val
        else:
            # Mode for categorical / string
            consensus[k] = max(set(values), key=values.count)

    return consensus
