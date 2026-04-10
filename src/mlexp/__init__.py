"""
mlexp — model-agnostic ML experimentation
==========================================

Quick-start
-----------
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
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
    search_strategy="grid",           # "grid" | "random" | "optuna"
    evaluation_strategy="nested_cv",  # "cv"   | "nested_cv" | "holdout"
    evaluation_kwargs={
        "outer_cv": StratifiedKFold(5, shuffle=True, random_state=42),
        "inner_cv": StratifiedKFold(3, shuffle=True, random_state=42),
    },
    fit_params={"sample_weight": weights},
)

exp = Experiment(log_dir="runs/")
result, record = exp.run(config, X_train, y_train)
print(f"{record.mean_score:.4f} ± {record.std_score:.4f}")
"""

from mlexp.config import ExperimentConfig
from mlexp.experiment import Experiment

__all__ = ["Experiment", "ExperimentConfig"]
