from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.model_selection import BaseCrossValidator

from mlexp.adapters.factory import wrap_model
from mlexp.search.base import BaseSearch, SearchResult
from mlexp.search.grid_search import _cv_scores


class OptunaSearch(BaseSearch):
    """
    Bayesian hyperparameter search via Optuna.

    param_space format — each key maps to a dict describing the suggest call:

        {
            "max_depth":    {"type": "int",        "low": 3, "high": 10},
            "learning_rate":{"type": "float",      "low": 1e-4, "high": 0.3, "log": True},
            "subsample":    {"type": "float",      "low": 0.5, "high": 1.0},
            "booster":      {"type": "categorical","choices": ["gbtree", "dart"]},
        }
    """

    def __init__(
        self,
        n_trials: int = 30,
        direction: str = "maximize",
        random_state: int | None = None,
        optuna_kwargs: dict | None = None,
    ) -> None:
        self.n_trials = n_trials
        self.direction = direction
        self.random_state = random_state
        self.optuna_kwargs = optuna_kwargs or {}

    def search(
        self,
        model_fn: Callable[..., Any],
        param_space: dict[str, dict],
        X: np.ndarray,
        y: np.ndarray,
        cv: BaseCrossValidator,
        score_fn: Callable,
        fit_params: dict[str, Any] | None = None,
    ) -> SearchResult:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        all_results: list[dict] = []

        def objective(trial: optuna.Trial) -> float:
            params = _suggest_params(trial, param_space)
            model = wrap_model(model_fn(**params))
            scores = _cv_scores(model, X, y, cv, score_fn, fit_params)
            mean = float(np.mean(scores))
            trial.set_user_attr("std", float(np.std(scores)))
            trial.set_user_attr("params", params)
            return mean

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
            **self.optuna_kwargs,
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        for t in study.trials:
            if t.value is not None:
                all_results.append(
                    {
                        "params": t.user_attrs.get("params", {}),
                        "mean": t.value,
                        "std": t.user_attrs.get("std", 0.0),
                    }
                )

        best = study.best_trial
        return SearchResult(
            best_params=best.user_attrs.get("params", {}),
            best_score=best.value,
            best_score_std=best.user_attrs.get("std", 0.0),
            all_results=all_results,
        )


def _suggest_params(trial, param_space: dict) -> dict:
    import optuna

    params: dict = {}
    for name, spec in param_space.items():
        ptype = spec["type"]
        if ptype == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"], **{k: v for k, v in spec.items() if k not in ("type", "low", "high")})
        elif ptype == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], **{k: v for k, v in spec.items() if k not in ("type", "low", "high")})
        elif ptype == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unknown param type '{ptype}' for '{name}'")
    return params
