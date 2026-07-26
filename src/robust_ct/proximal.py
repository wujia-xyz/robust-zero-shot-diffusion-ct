"""Framework-independent warm-started proximal conjugate gradients."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


Array = NDArray[np.float64]
LinearMap = Callable[[Array], ArrayLike]


@dataclass(frozen=True)
class CGResult:
    """Solution and diagnostics from a proximal-CG update."""

    solution: Array
    iterations: int
    converged: bool
    residual_norm: float


def _inner(left: Array, right: Array) -> float:
    return float(np.vdot(left.reshape(-1), right.reshape(-1)).real)


def proximal_cg(
    forward: LinearMap,
    adjoint: LinearMap,
    measurement: ArrayLike,
    prior_center: ArrayLike,
    prior_weight: float,
    *,
    iterations: int = 6,
    weights: ArrayLike | None = None,
    initial: ArrayLike | None = None,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> CGResult:
    """Solve the weighted proximal normal equation with conjugate gradients.

    The solved system is

    ``(A.T @ W @ A + gamma I) x = A.T @ W @ y + gamma x_prior``.

    The default initial iterate is ``prior_center``, matching the warm start
    used by the reconstruction trajectory.  ``forward`` and ``adjoint`` may
    wrap any numerical backend as long as they accept and return array-like
    objects of the corresponding image and measurement shapes.
    """

    if isinstance(iterations, bool) or int(iterations) != iterations or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer.")
    iterations = int(iterations)
    prior_weight = float(prior_weight)
    if not math.isfinite(prior_weight) or prior_weight <= 0.0:
        raise ValueError("prior_weight must be finite and positive.")
    for value, name in ((rtol, "rtol"), (atol, "atol")):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative.")

    dtype = np.result_type(measurement, prior_center, np.float64)
    y = np.asarray(measurement, dtype=dtype)
    center = np.asarray(prior_center, dtype=dtype)
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(center)):
        raise ValueError("measurement and prior_center must contain finite values.")

    if weights is None:
        weight_array: NDArray[np.floating] | None = None
    else:
        raw_weights = np.asarray(weights, dtype=dtype)
        try:
            weight_array = np.broadcast_to(raw_weights, y.shape)
        except ValueError as exc:
            raise ValueError("weights must be broadcastable to measurement.shape.") from exc
        if not np.all(np.isfinite(weight_array)) or np.any(weight_array < 0.0):
            raise ValueError("weights must be finite and nonnegative.")

    def apply_forward(image: Array) -> Array:
        projected = np.asarray(forward(image), dtype=dtype)
        if projected.shape != y.shape:
            raise ValueError(
                "forward(image) must have measurement.shape; "
                f"got {projected.shape}, expected {y.shape}."
            )
        return projected

    def apply_adjoint(projected: Array) -> Array:
        image = np.asarray(adjoint(projected), dtype=dtype)
        if image.shape != center.shape:
            raise ValueError(
                "adjoint(measurement) must have prior_center.shape; "
                f"got {image.shape}, expected {center.shape}."
            )
        return image

    def apply_weights(projected: Array) -> Array:
        return projected if weight_array is None else weight_array * projected

    def normal(image: Array) -> Array:
        return apply_adjoint(apply_weights(apply_forward(image))) + prior_weight * image

    rhs = apply_adjoint(apply_weights(y)) + prior_weight * center
    x = (
        center.copy()
        if initial is None
        else np.asarray(initial, dtype=dtype).copy()
    )
    if x.shape != center.shape:
        raise ValueError("initial must have prior_center.shape.")
    if not np.all(np.isfinite(x)):
        raise ValueError("initial must contain finite values.")

    residual = rhs - normal(x)
    direction = residual.copy()
    residual_sq = _inner(residual, residual)
    initial_norm = math.sqrt(max(residual_sq, 0.0))
    numerical_floor = np.finfo(dtype).eps * max(1.0, initial_norm)
    tolerance = max(float(atol), float(rtol) * initial_norm, numerical_floor)
    if initial_norm <= tolerance or iterations == 0:
        return CGResult(
            solution=x,
            iterations=0,
            converged=initial_norm <= tolerance,
            residual_norm=initial_norm,
        )

    completed = 0
    converged = False
    for step in range(iterations):
        normal_direction = normal(direction)
        denominator = _inner(direction, normal_direction)
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise np.linalg.LinAlgError(
                "CG encountered a non-positive curvature direction; "
                "check that A.T is the adjoint and W is nonnegative."
            )
        alpha = residual_sq / denominator
        x += alpha * direction
        residual -= alpha * normal_direction
        new_residual_sq = max(_inner(residual, residual), 0.0)
        completed = step + 1
        residual_norm = math.sqrt(new_residual_sq)
        if residual_norm <= tolerance:
            converged = True
            residual_sq = new_residual_sq
            break
        beta = new_residual_sq / residual_sq
        direction = residual + beta * direction
        residual_sq = new_residual_sq

    return CGResult(
        solution=x,
        iterations=completed,
        converged=converged,
        residual_norm=math.sqrt(max(residual_sq, 0.0)),
    )
