"""Detector-column reliability estimation from a measured sinogram."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class DetectorReliability:
    """Intermediate statistics and the resulting detector-column weights."""

    profile: NDArray[np.float64]
    trend: NDArray[np.float64]
    residual: NDArray[np.float64]
    scaled_mad: float
    reliable: NDArray[np.bool_]
    weights: NDArray[np.float64]


def _running_median_reflect(
    values: NDArray[np.float64],
    width: int,
) -> NDArray[np.float64]:
    half_width = width // 2
    padded = np.pad(values, (half_width, half_width), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    return np.median(windows, axis=-1)


def detector_column_reliability(
    sinogram: ArrayLike,
    *,
    width: int = 11,
    threshold: float = 3.0,
    mad_scale: float = 1.4826,
    epsilon: float = 1.0e-12,
) -> DetectorReliability:
    """Detect unreliable detector columns using an angle-averaged MAD rule.

    The angle-averaged detector profile is compared with a reflect-padded
    running-median trend.  A column is unreliable when its residual differs
    from the median residual by more than ``threshold`` scaled MADs.  Returned
    weights are one for reliable columns and zero otherwise.
    """

    array = np.asarray(sinogram, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("sinogram must have shape (projection_angles, detectors).")
    if min(array.shape) < 1 or not np.all(np.isfinite(array)):
        raise ValueError("sinogram must be nonempty and contain finite values.")
    if isinstance(width, bool) or int(width) != width or width < 3 or width % 2 == 0:
        raise ValueError("width must be an odd integer of at least 3.")
    width = int(width)
    if width > array.shape[1]:
        raise ValueError("width cannot exceed the number of detector columns.")
    threshold = float(threshold)
    mad_scale = float(mad_scale)
    epsilon = float(epsilon)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold must be finite and positive.")
    if not math.isfinite(mad_scale) or mad_scale <= 0.0:
        raise ValueError("mad_scale must be finite and positive.")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive.")

    profile = np.mean(array, axis=0)
    trend = _running_median_reflect(profile, width)
    residual = profile - trend
    residual_median = float(np.median(residual))
    deviations = np.abs(residual - residual_median)
    scaled_mad = mad_scale * float(np.median(deviations)) + epsilon
    reliable = deviations <= threshold * scaled_mad
    weights = reliable.astype(np.float64)
    return DetectorReliability(
        profile=profile,
        trend=trend,
        residual=residual,
        scaled_mad=scaled_mad,
        reliable=reliable,
        weights=weights,
    )
