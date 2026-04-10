# mlexp

Model-agnostic machine learning experimentation for tabular models. `mlexp` unifies hyperparameter search, evaluation, distance-based sample weighting, and lightweight experiment tracking in a small, opinionated library.

## Overview

`mlexp` is built around three core ideas:

- **Search**: support grid search, random search, and Optuna-based tuning.
- **Evaluation**: support k-fold CV, nested CV, and holdout evaluation strategies.
- **Adapters**: support any sklearn-compatible model plus CatBoost with automatic wrapper handling.

The package is designed for fast experimentation on tabular data with tree models and other sklearn-style estimators.

## Key Concepts

### Experiment orchestration

The top-level API is `Experiment` together with `ExperimentConfig`.

- `Experiment` orchestrates search + evaluation + logging.
- `ExperimentConfig` defines the model factory, parameter space, score function, strategy choices, and optional metadata.

The `Experiment.run()` method returns both the raw evaluation result and a persisted `ExperimentRecord`.

### Model adapters

`mlexp` wraps models with a lightweight adapter abstraction:

- `SklearnAdapter` for sklearn-compatible estimators and wrappers.
- `CatBoostAdapter` for CatBoost models with a CatBoost-specific fit signature.

This allows consistent `fit`, `predict`, and `predict_proba` behavior across model types.

### Search strategies

The package supports three pluggable searchers:

- `grid` — exhaustive parameter grid search.
- `random` — randomized sampling over a search space.
- `optuna` — Bayesian search using Optuna.

Each search strategy returns a best parameter set plus the full search history.

### Evaluation strategies

`mlexp` supports:

- `cv` — simple cross-validation with hyperparameter tuning on the entire dataset.
- `nested_cv` — unbiased nested cross-validation with separate tuning and outer evaluation folds.
- `holdout` — tune on a training split and validate on a fixed holdout set.

`nested_cv` is especially useful when you want a reliable estimate of generalization performance.

### Consensus parameters

When nested CV returns one best parameter set per outer fold, `Experiment` derives a single consensus configuration:

- numeric values: median across fold-optimal values.
- categorical values: mode of the fold choices.

This gives a stable, deployable parameter set after nested evaluation.

### Distance-based sample weighting

`mlexp` includes the `DistanceWeighter` utility for training-time sample weighting.

- Compute weights for each training sample based on similarity to a reference set.
- Supports `rbf`, `cosine`, and `euclidean` distance metrics.
- Optionally normalizes weights so the mean weight equals `1`.

This is useful for correcting distribution shift or emphasizing examples that are closer to a target distribution.

### Experiment logging

`ExperimentLogger` persists every completed run as a JSON line in `mlexp_runs/runs.jsonl`.

Each run record contains:

- `run_id`
- `model_name`
- `search_strategy`
- `evaluation_strategy`
- `best_params`
- `mean_score`, `std_score`
- `outer_scores`
- `best_params_per_fold`
- optional additional metric summaries
- tags and timestamp

## Installation

```bash
pip install .
```

This package depends on:

- Python 3.11+
- `numpy`
- `pandas`
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `catboost`
- `optuna`

For development and testing:

```bash
pip install -r requirements-dev.txt
```

## Quickstart

```python
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
```

## Configuration reference

### `ExperimentConfig`

- `model_fn`: callable returning a model instance, e.g. `lambda **p: XGBClassifier(**p)`.
- `model_name`: human-friendly name used for logging.
- `param_space`: search space for hyperparameters.
- `score_fn`: primary selection metric, signature `(adapter, X, y) -> float`.
- `additional_metrics`: optional dict of extra metrics for reporting only.
- `search_strategy`: one of `grid`, `random`, `optuna`.
- `search_kwargs`: forwarded to the chosen searcher constructor.
- `evaluation_strategy`: one of `cv`, `nested_cv`, `holdout`.
- `evaluation_kwargs`: forwarded to the chosen evaluator.
- `fit_params`: optional args passed to `model.fit()`, e.g. `sample_weight`.
- `tags`: optional metadata recorded with the run.
- `run_id`: optional string to override automatic run IDs.

### Search parameter formats

- `grid`: dictionary of discrete lists.
- `random`: values may be lists, `scipy` distributions, or callables.
- `optuna`: each key must be a spec dict with `type` plus bounds or choices.

Example Optuna space:

```python
param_space = {
    "max_depth": {"type": "int", "low": 3, "high": 10},
    "learning_rate": {"type": "float", "low": 1e-4, "high": 0.3, "log": True},
    "booster": {"type": "categorical", "choices": ["gbtree", "dart"]},
}
```

### Evaluation kwargs

- `cv`: any `sklearn.model_selection.BaseCrossValidator`.
- `outer_cv` / `inner_cv`: used by `nested_cv`.
- `X_holdout`, `y_holdout`, `cv`: used by `holdout`.

## Example: additional metrics

```python
from sklearn.metrics import roc_auc_score, f1_score

config = ExperimentConfig(
    ...
    additional_metrics={
        "roc_auc": lambda a, X, y: roc_auc_score(y, a.predict_proba(X)[:, 1]),
        "f1": lambda a, X, y: f1_score(y, a.predict(X)),
    },
)
```

Additional metrics are computed only for reporting and do not affect hyperparameter selection.

## Example: holdout evaluation

```python
from sklearn.model_selection import StratifiedKFold

config = ExperimentConfig(
    ...
    evaluation_strategy="holdout",
    evaluation_kwargs={
        "X_holdout": X_test,
        "y_holdout": y_test,
        "cv": StratifiedKFold(3, shuffle=True, random_state=42),
    },
)
```

## Distance weighting

Use `compute_distance_weights()` to produce sample weights from training data and a reference distribution:

```python
from mlexp.weighting import compute_distance_weights

weights = compute_distance_weights(
    X_train,
    X_reference,
    metric="rbf",
    bandwidth=1.0,
    normalize=True,
    clip_percentile=99.0,
)
```

Then pass `weights` in `fit_params`:

```python
config = ExperimentConfig(
    ...
    fit_params={"sample_weight": weights},
)
```

## Logging and experiment tracking

`Experiment` writes records to `mlexp_runs/runs.jsonl` by default.

The logger can also read back runs and identify the best recorded model by mean score.

## Notes and best practices

- `score_fn` is the only metric used for hyperparameter selection. Keep it focused on your target objective.
- `additional_metrics` are for reporting only and should not affect tuning.
- Use `nested_cv` when you need an unbiased estimate of generalization performance.
- Use the consensus parameter logic after nested CV to derive a stable final parameter set.
- `DistanceWeighter` is useful when your training and inference distributions differ.

## Project structure

- `src/mlexp/` — package modules
  - `config.py` — experiment configuration dataclass
  - `experiment.py` — orchestration and logging
  - `adapters/` — model wrapper logic
  - `evaluation/` — CV, nested CV, and holdout evaluators
  - `search/` — grid, random, and Optuna search implementations
  - `tracking/` — experiment logger and record model
  - `training/` — self-training utilities (future/extended support)
  - `weighting/` — distance-based sample weighting utilities

## License

This repository is currently maintained as part of an internal ML experimentation toolkit.
