from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity


class DistanceWeighter:
    """
    Compute sample weights based on distance/similarity to a reference
    distribution or set of reference points.

    The idea (from the distance-based weighting paper): training samples that
    are more similar to what the model will see at inference time should
    receive higher weight during training.  This corrects for distribution
    shift and downweights noisy/distant examples naturally.

    Parameters
    ----------
    metric : "rbf" | "cosine" | "euclidean"
        Similarity/distance function.
    bandwidth : float
        Bandwidth parameter for RBF kernel (γ in exp(-γ ‖x-y‖²)).
        Only used when metric="rbf".
    normalize : bool
        If True, weights are normalised so that mean(weights) == 1.
        This keeps the effective sample size stable and prevents unstable
        training from very large or very small weight values.
    clip_percentile : float | None
        If set, weights above this percentile are clipped.  Useful to
        prevent a handful of very close reference points from dominating.
    """

    def __init__(
        self,
        metric: Literal["rbf", "cosine", "euclidean"] = "rbf",
        bandwidth: float = 1.0,
        normalize: bool = True,
        clip_percentile: float | None = 99.0,
    ) -> None:
        self.metric = metric
        self.bandwidth = bandwidth
        self.normalize = normalize
        self.clip_percentile = clip_percentile

    def fit(self, X_reference: np.ndarray) -> "DistanceWeighter":
        """Store reference distribution (e.g. test/target set)."""
        self.X_reference_ = np.asarray(X_reference, dtype=float)
        return self

    def transform(self, X_train: np.ndarray) -> np.ndarray:
        """
        Return a weight array of shape (n_train,).

        Each weight reflects how similar a training sample is to the
        reference distribution on average.
        """
        X_train = np.asarray(X_train, dtype=float)
        similarities = self._pairwise_similarity(X_train, self.X_reference_)
        # Average similarity of each train sample to ALL reference samples
        weights = similarities.mean(axis=1)
        return self._postprocess(weights)

    def fit_transform(
        self, X_reference: np.ndarray, X_train: np.ndarray
    ) -> np.ndarray:
        return self.fit(X_reference).transform(X_train)

    # ------------------------------------------------------------------
    def _pairwise_similarity(
        self, X: np.ndarray, Y: np.ndarray
    ) -> np.ndarray:
        if self.metric == "rbf":
            return rbf_kernel(X, Y, gamma=self.bandwidth)
        if self.metric == "cosine":
            return (cosine_similarity(X, Y) + 1.0) / 2.0  # shift to [0,1]
        if self.metric == "euclidean":
            from sklearn.metrics.pairwise import euclidean_distances
            dists = euclidean_distances(X, Y)
            # Convert distance → similarity with Gaussian kernel
            return np.exp(-self.bandwidth * dists**2)
        raise ValueError(f"Unknown metric '{self.metric}'")

    def _postprocess(self, weights: np.ndarray) -> np.ndarray:
        weights = weights.copy()

        if self.clip_percentile is not None:
            cap = np.percentile(weights, self.clip_percentile)
            weights = np.clip(weights, 0, cap)

        # Avoid zero weights entirely
        weights = np.maximum(weights, 1e-8)

        if self.normalize:
            weights = weights / weights.mean()

        return weights


def compute_distance_weights(
    X_train: np.ndarray,
    X_reference: np.ndarray,
    metric: Literal["rbf", "cosine", "euclidean"] = "rbf",
    bandwidth: float = 1.0,
    normalize: bool = True,
    clip_percentile: float | None = 99.0,
) -> np.ndarray:
    """
    Convenience function wrapping DistanceWeighter.fit_transform().

    Returns weight array of shape (len(X_train),) ready to be passed as
    sample_weight to any model's fit() method.
    """
    return DistanceWeighter(
        metric=metric,
        bandwidth=bandwidth,
        normalize=normalize,
        clip_percentile=clip_percentile,
    ).fit_transform(X_reference, X_train)
