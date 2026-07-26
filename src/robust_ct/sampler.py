"""Framework-neutral reverse sampler for the acquisition-conditioned law.

This is an original, compact reference implementation of the paper's numerical
loop.  It is not copied from, and does not import, the upstream DM4CT sampler.
Callers supply their own denoiser and forward/adjoint operators.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .law import DEFAULT_DENOMINATOR_EPSILON
from .proximal import LinearMap, proximal_cg
from .schedule import prior_anchor_trajectory


Array = NDArray[np.float64]
Denoiser = Callable[[Array, float], ArrayLike]
Projection = Callable[[Array], ArrayLike]
Transition = Literal["renoise", "ode"]


@dataclass(frozen=True)
class SamplerResult:
    """Ensemble reconstruction, individual chains, and shared anchor weights."""

    reconstruction: Array
    chains: Array
    prior_anchor_weights: Array


def _identity(array: Array) -> Array:
    return array


def reconstruct(
    measurement: ArrayLike,
    forward: LinearMap,
    adjoint: LinearMap,
    denoise: Denoiser,
    sigma_levels: ArrayLike,
    *,
    gain: float,
    gamma_bar: float,
    cap: float,
    cg_iterations: int = 6,
    chain_count: int = 1,
    weights: ArrayLike | None = None,
    initial_mean: ArrayLike | None = None,
    transition: Transition = "renoise",
    seed: int | None = None,
    clean_projection: Projection | None = None,
    dc_projection: Projection | None = None,
    denominator_epsilon: float = DEFAULT_DENOMINATOR_EPSILON,
) -> SamplerResult:
    """Run the law-conditioned proximal-CG reverse loop.

    ``sigma_levels`` must contain descending positive levels followed by zero.
    In ``"renoise"`` mode, each conditioned clean estimate is perturbed by
    independent isotropic Gaussian noise at the next level.  ``"ode"`` uses a
    first-order VE probability-flow update with the conditioned clean estimate
    as the denoising target.
    """

    y = np.asarray(measurement, dtype=np.float64)
    if y.size == 0 or not np.all(np.isfinite(y)):
        raise ValueError("measurement must be nonempty and finite.")
    sigmas = np.asarray(sigma_levels, dtype=np.float64)
    if sigmas.ndim != 1 or sigmas.size < 2:
        raise ValueError("sigma_levels must be a 1-D array with at least two entries.")
    if not np.all(np.isfinite(sigmas)):
        raise ValueError("sigma_levels must be finite.")
    if sigmas[-1] != 0.0 or np.any(sigmas[:-1] <= 0.0):
        raise ValueError("sigma_levels must contain positive levels followed by zero.")
    if np.any(np.diff(sigmas) >= 0.0):
        raise ValueError("sigma_levels must be strictly descending.")
    if transition not in ("renoise", "ode"):
        raise ValueError("transition must be either 'renoise' or 'ode'.")
    if isinstance(chain_count, bool) or int(chain_count) != chain_count or chain_count < 1:
        raise ValueError("chain_count must be a positive integer.")
    chain_count = int(chain_count)
    for value, name, allow_zero in (
        (gain, "gain", False),
        (gamma_bar, "gamma_bar", False),
        (cap, "cap", False),
        (denominator_epsilon, "denominator_epsilon", True),
    ):
        value = float(value)
        valid = value >= 0.0 if allow_zero else value > 0.0
        if not math.isfinite(value) or not valid:
            qualifier = "nonnegative" if allow_zero else "positive"
            raise ValueError(f"{name} must be finite and {qualifier}.")

    backprojected = np.asarray(adjoint(y), dtype=np.float64)
    if backprojected.size == 0 or not np.all(np.isfinite(backprojected)):
        raise ValueError("adjoint(measurement) must be nonempty and finite.")
    if initial_mean is None:
        mean = np.zeros_like(backprojected)
    else:
        mean = np.asarray(initial_mean, dtype=np.float64)
        if mean.shape != backprojected.shape or not np.all(np.isfinite(mean)):
            raise ValueError(
                "initial_mean must be finite and have adjoint(measurement).shape."
            )

    clean_projection = clean_projection or _identity
    dc_projection = dc_projection or _identity
    anchor_weights = prior_anchor_trajectory(
        sigmas[:-1],
        gamma_bar,
        cap,
        gain,
        denominator_epsilon=denominator_epsilon,
    )
    generator = np.random.default_rng(seed)
    final_chains: list[Array] = []

    for _ in range(chain_count):
        state = mean + sigmas[0] * generator.standard_normal(mean.shape)
        for index, sigma in enumerate(sigmas[:-1]):
            sigma_next = float(sigmas[index + 1])
            clean = np.asarray(denoise(state, float(sigma)), dtype=np.float64)
            if clean.shape != mean.shape or not np.all(np.isfinite(clean)):
                raise ValueError(
                    "denoise(state, sigma) must return a finite array with image shape."
                )
            clean = np.asarray(clean_projection(clean), dtype=np.float64)
            if clean.shape != mean.shape or not np.all(np.isfinite(clean)):
                raise ValueError("clean_projection returned an invalid image.")
            dc_result = proximal_cg(
                forward,
                adjoint,
                y,
                clean,
                float(anchor_weights[index]),
                iterations=cg_iterations,
                weights=weights,
                initial=clean,
            )
            conditioned = np.asarray(dc_projection(dc_result.solution), dtype=np.float64)
            if conditioned.shape != mean.shape or not np.all(np.isfinite(conditioned)):
                raise ValueError("dc_projection returned an invalid image.")

            if transition == "renoise":
                state = conditioned
                if sigma_next > 0.0:
                    state = state + sigma_next * generator.standard_normal(mean.shape)
            else:
                state = state + (sigma_next - sigma) * (
                    state - conditioned
                ) / sigma
        final_chains.append(np.asarray(state, dtype=np.float64))

    chains = np.stack(final_chains, axis=0)
    return SamplerResult(
        reconstruction=np.mean(chains, axis=0),
        chains=chains,
        prior_anchor_weights=anchor_weights,
    )
