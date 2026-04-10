from __future__ import annotations

from mlexp.search.base import BaseSearch
from mlexp.search.grid_search import GridSearch
from mlexp.search.optuna_search import OptunaSearch
from mlexp.search.random_search import RandomSearch

_REGISTRY: dict[str, type[BaseSearch]] = {
    "grid": GridSearch,
    "random": RandomSearch,
    "optuna": OptunaSearch,
}


def get_searcher(strategy: str, **kwargs) -> BaseSearch:
    """
    Factory for search strategies.

    strategy: "grid" | "random" | "optuna"
    kwargs  : forwarded to the strategy constructor
    """
    key = strategy.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown search strategy '{strategy}'. Choose from: {list(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)
