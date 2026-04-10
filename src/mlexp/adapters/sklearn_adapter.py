from __future__ import annotations

import inspect

import numpy as np

from mlexp.adapters.base import ModelAdapter


class SklearnAdapter(ModelAdapter):
    """
    Thin wrapper for any sklearn-compatible model (sklearn, XGBoost, LightGBM,
    CatBoost sklearn wrapper).  Handles the sample_weight dispatch gracefully:
    if the model's fit() doesn't accept sample_weight it is silently dropped.
    """

    def __init__(self, model) -> None:
        self._model = model

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "SklearnAdapter":
        if sample_weight is not None and self._accepts_sample_weight():
            self._model.fit(X, y, sample_weight=sample_weight)
        else:
            self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self._model, "predict_proba"):
            raise NotImplementedError(
                f"{type(self._model).__name__} does not support predict_proba"
            )
        return self._model.predict_proba(X)

    @property
    def raw_model(self):
        return self._model

    # ------------------------------------------------------------------
    def _accepts_sample_weight(self) -> bool:
        try:
            sig = inspect.signature(self._model.fit)
            return "sample_weight" in sig.parameters
        except (ValueError, TypeError):
            return False


class CatBoostAdapter(SklearnAdapter):
    """
    CatBoost-specific adapter.  CatBoost's fit() uses a slightly different
    signature and benefits from verbose=0 to suppress stdout.
    """

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "CatBoostAdapter":
        kwargs: dict = {"verbose": 0}
        if sample_weight is not None:
            kwargs["sample_weight"] = sample_weight
        self._model.fit(X, y, **kwargs)
        return self
