from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ModelAdapter(ABC):
    """Minimal contract all model wrappers must satisfy."""

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "ModelAdapter":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            f"{type(self).__name__} does not support predict_proba"
        )

    @property
    @abstractmethod
    def raw_model(self):
        """Return the underlying model object."""
        ...
