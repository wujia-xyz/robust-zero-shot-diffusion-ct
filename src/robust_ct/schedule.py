"""Noise and prior-anchor schedules used by the reconstruction sampler."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .law import DEFAULT_DENOMINATOR_EPSILON


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive; got {value!r}.")
    return value


def power_noise_schedule(
    num_positive_levels: int,
    sigma_max: float,
    sigma_min: float = 0.01,
    *,
    power: float = 7.0,
    append_zero: bool = True,
) -> NDArray[np.float64]:
    """Construct a descending power schedule in the equivalent VE coordinate.

    The positive levels interpolate linearly between
    ``sigma_max**(1/power)`` and ``sigma_min**(1/power)`` before being raised
    back to ``power``.  A final zero is appended by default.
    """

    if isinstance(num_positive_levels, bool) or num_positive_levels < 2:
        raise ValueError("num_positive_levels must be an integer of at least 2.")
    if int(num_positive_levels) != num_positive_levels:
        raise ValueError("num_positive_levels must be an integer.")
    sigma_max = _positive_finite(sigma_max, "sigma_max")
    sigma_min = _positive_finite(sigma_min, "sigma_min")
    power = _positive_finite(power, "power")
    if sigma_max <= sigma_min:
        raise ValueError("sigma_max must be greater than sigma_min.")

    ramp = np.linspace(0.0, 1.0, int(num_positive_levels), dtype=np.float64)
    root_max = sigma_max ** (1.0 / power)
    root_min = sigma_min ** (1.0 / power)
    levels = (root_max + ramp * (root_min - root_max)) ** power
    if append_zero:
        levels = np.concatenate((levels, np.zeros(1, dtype=np.float64)))
    return levels


def select_two_regime_gamma_bar(
    noise_variance: float,
    *,
    noise_free_value: float = 0.3,
    noisy_value: float = 0.1,
) -> float:
    """Apply the pre-specified noise-free/noisy backbone rule."""

    noise_variance = float(noise_variance)
    if not math.isfinite(noise_variance) or noise_variance < 0.0:
        raise ValueError("noise_variance must be finite and nonnegative.")
    noise_free_value = _positive_finite(noise_free_value, "noise_free_value")
    noisy_value = _positive_finite(noisy_value, "noisy_value")
    return noise_free_value if noise_variance == 0.0 else noisy_value


def normalized_anchor_trajectory(
    positive_sigma_levels: ArrayLike,
    gamma_bar: float,
    cap: float,
    *,
    denominator_epsilon: float = DEFAULT_DENOMINATOR_EPSILON,
) -> NDArray[np.float64]:
    """Return ``min(gamma_bar/(sigma**2 + eps), cap)`` at each level."""

    sigmas = np.asarray(positive_sigma_levels, dtype=np.float64)
    if sigmas.ndim != 1 or sigmas.size == 0:
        raise ValueError("positive_sigma_levels must be a nonempty 1-D array.")
    if not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0.0):
        raise ValueError("All positive_sigma_levels must be finite and positive.")
    gamma_bar = float(gamma_bar)
    cap = float(cap)
    denominator_epsilon = float(denominator_epsilon)
    if not math.isfinite(gamma_bar) or gamma_bar < 0.0:
        raise ValueError("gamma_bar must be finite and nonnegative.")
    if not math.isfinite(cap) or cap < 0.0:
        raise ValueError("cap must be finite and nonnegative.")
    if not math.isfinite(denominator_epsilon) or denominator_epsilon < 0.0:
        raise ValueError("denominator_epsilon must be finite and nonnegative.")
    return np.minimum(
        gamma_bar / (sigmas**2 + denominator_epsilon),
        cap,
    )


def prior_anchor_trajectory(
    positive_sigma_levels: ArrayLike,
    gamma_bar: float,
    cap: float,
    gain: float,
    *,
    denominator_epsilon: float = DEFAULT_DENOMINATOR_EPSILON,
) -> NDArray[np.float64]:
    """Return the physical proximal weights ``GAIN * normalized_anchor``."""

    gain = _positive_finite(gain, "gain")
    return gain * normalized_anchor_trajectory(
        positive_sigma_levels,
        gamma_bar,
        cap,
        denominator_epsilon=denominator_epsilon,
    )
