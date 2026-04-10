"""
End-to-end smoke tests.  Fast, no heavy training.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from mlexp import Experiment, ExperimentConfig
from mlexp.adapters import wrap_model
from mlexp.evaluation import get_evaluator
from mlexp.search import get_searcher
from mlexp.weighting import compute_distance_weights, DistanceWeighter


@pytest.fixture
def toy_data():
    X, y = make_classification(n_samples=200, n_features=10, random_state=0)
    return X.astype(float), y


def score_fn(adapter, X, y):
    return accuracy_score(y, adapter.predict(X))


def model_fn(**params):
    return LogisticRegression(**params, max_iter=500)


# ------------------------------------------------------------------
class TestAdapters:
    def test_wrap_sklearn(self, toy_data):
        X, y = toy_data
        adapter = wrap_model(model_fn(C=1.0))
        adapter.fit(X, y)
        preds = adapter.predict(X)
        assert len(preds) == len(y)

    def test_sample_weight_passthrough(self, toy_data):
        X, y = toy_data
        sw = np.ones(len(y))
        adapter = wrap_model(model_fn(C=1.0))
        adapter.fit(X, y, sample_weight=sw)
        assert adapter.predict(X).shape == y.shape


# ------------------------------------------------------------------
class TestSearchStrategies:
    def test_grid_search(self, toy_data):
        X, y = toy_data
        searcher = get_searcher("grid")
        cv = StratifiedKFold(3, shuffle=True, random_state=0)
        result = searcher.search(model_fn, {"C": [0.1, 1.0]}, X, y, cv, score_fn)
        assert result.best_params["C"] in [0.1, 1.0]
        assert 0.0 < result.best_score <= 1.0

    def test_random_search(self, toy_data):
        X, y = toy_data
        searcher = get_searcher("random", n_iter=5, random_state=42)
        cv = StratifiedKFold(3, shuffle=True, random_state=0)
        result = searcher.search(model_fn, {"C": [0.1, 0.5, 1.0, 2.0]}, X, y, cv, score_fn)
        assert "C" in result.best_params

    def test_optuna_search(self, toy_data):
        X, y = toy_data
        searcher = get_searcher("optuna", n_trials=5, random_state=0)
        cv = StratifiedKFold(3, shuffle=True, random_state=0)
        param_space = {"C": {"type": "float", "low": 0.01, "high": 10.0, "log": True}}
        result = searcher.search(model_fn, param_space, X, y, cv, score_fn)
        assert "C" in result.best_params


# ------------------------------------------------------------------
class TestEvaluationStrategies:
    def test_cv_evaluator(self, toy_data):
        X, y = toy_data
        evaluator = get_evaluator("cv", cv=StratifiedKFold(3, shuffle=True, random_state=0))
        searcher = get_searcher("grid")
        result = evaluator.evaluate(model_fn, {"C": [0.1, 1.0]}, X, y, searcher, score_fn)
        assert len(result.outer_scores) == 3

    def test_nested_cv_evaluator(self, toy_data):
        X, y = toy_data
        evaluator = get_evaluator(
            "nested_cv",
            outer_cv=StratifiedKFold(3, shuffle=True, random_state=0),
            inner_cv=StratifiedKFold(2, shuffle=True, random_state=0),
        )
        searcher = get_searcher("grid")
        result = evaluator.evaluate(model_fn, {"C": [0.1, 1.0]}, X, y, searcher, score_fn)
        assert len(result.outer_scores) == 3
        assert len(result.best_params_per_fold) == 3

    def test_holdout_evaluator(self, toy_data):
        X, y = toy_data
        split = 160
        evaluator = get_evaluator(
            "holdout",
            X_holdout=X[split:],
            y_holdout=y[split:],
            cv=StratifiedKFold(3, shuffle=True, random_state=0),
        )
        searcher = get_searcher("grid")
        result = evaluator.evaluate(model_fn, {"C": [0.1, 1.0]}, X[:split], y[:split], searcher, score_fn)
        assert len(result.outer_scores) == 1


# ------------------------------------------------------------------
class TestWeighting:
    def test_distance_weights_shape(self, toy_data):
        X, y = toy_data
        weights = compute_distance_weights(X[:100], X_reference=X[100:])
        assert weights.shape == (100,)
        assert (weights > 0).all()

    def test_normalized_weights_mean(self, toy_data):
        X, y = toy_data
        weights = compute_distance_weights(X[:100], X_reference=X[100:], normalize=True)
        assert abs(weights.mean() - 1.0) < 1e-6

    def test_cosine_metric(self, toy_data):
        X, y = toy_data
        w = DistanceWeighter(metric="cosine").fit_transform(X[100:], X[:100])
        assert w.shape == (100,)


# ------------------------------------------------------------------
class TestExperiment:
    def test_full_run(self, toy_data, tmp_path):
        X, y = toy_data
        config = ExperimentConfig(
            model_fn=model_fn,
            model_name="logreg",
            param_space={"C": [0.1, 1.0]},
            score_fn=score_fn,
            search_strategy="grid",
            evaluation_strategy="cv",
            evaluation_kwargs={"cv": StratifiedKFold(3, shuffle=True, random_state=0)},
        )
        exp = Experiment(log_dir=tmp_path)
        result, record = exp.run(config, X, y)
        assert record.model_name == "logreg"
        assert 0.0 < record.mean_score <= 1.0
        assert (tmp_path / "runs.jsonl").exists()
