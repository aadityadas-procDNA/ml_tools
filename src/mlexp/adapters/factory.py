from __future__ import annotations

from mlexp.adapters.base import ModelAdapter
from mlexp.adapters.sklearn_adapter import CatBoostAdapter, SklearnAdapter


def wrap_model(model) -> ModelAdapter:
    """
    Return the appropriate adapter for a model instance.

    Already-adapted models are returned as-is.  CatBoost models get a
    dedicated adapter; everything else gets the generic sklearn adapter.
    """
    if isinstance(model, ModelAdapter):
        return model

    module = type(model).__module__ or ""
    if "catboost" in module.lower():
        return CatBoostAdapter(model)

    return SklearnAdapter(model)
