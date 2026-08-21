"""Three-state own-slice reverse update used by the proposed method.

The module is intentionally backend neutral at the CT boundary: callers supply
the frozen two-dimensional prior, three forward/adjoint or DC callbacks, and
three physical measurements.  No neighboring measurement is passed to the
center data-consistency solve.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Protocol, Sequence

import torch

from .axial import CENTER_INDEX, PHYSICAL_SLICES, normalize_x0


EPSILON = 1.0e-8
SLICE_LABELS = ("z-1", "z", "z+1")

Tensor = torch.Tensor
BaseForward = Callable[[Tensor, Tensor], Tensor]
DCSolver = Callable[[Tensor, Tensor, float, int], Tensor]
Projection = Callable[[Tensor], Tensor]
EpsilonBridge = Callable[[Tensor, Tensor, Tensor], Tensor]


def _require_finite(name: str, value: Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")


def _require_triplet(name: str, value: Tensor) -> tuple[int, int, int]:
    if value.ndim != 5 or tuple(value.shape[1:3]) != (PHYSICAL_SLICES, 1):
        raise ValueError(
            f"{name} must have shape [K,3,1,H,W], got {tuple(value.shape)}"
        )
    if value.shape[0] < 1 or min(value.shape[-2:]) < 1:
        raise ValueError(f"{name} has an empty dimension")
    _require_finite(name, value)
    return int(value.shape[0]), int(value.shape[-2]), int(value.shape[-1])


@dataclass(frozen=True)
class SliceObservation:
    """Measurement belonging to one physical state in the triplet."""

    measurement: Tensor
    label: str


@dataclass(frozen=True)
class TripletSamplerConfig:
    """Numerical settings for acquisition-conditioned triplet sampling."""

    num_levels: int
    cg_iterations: int
    gamma_bar: float
    g_cap: float
    gain: float
    lower: float
    upper: float
    ensemble_size: int = 1
    denominator_epsilon: float = EPSILON

    def validate(self) -> None:
        if self.num_levels < 1 or self.cg_iterations < 1:
            raise ValueError("num_levels and cg_iterations must be positive")
        if self.ensemble_size != 1:
            raise ValueError("the released method uses one stochastic trajectory")
        for name, value in (
            ("gamma_bar", self.gamma_bar),
            ("g_cap", self.g_cap),
            ("gain", self.gain),
            ("lower", self.lower),
            ("upper", self.upper),
            ("denominator_epsilon", self.denominator_epsilon),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.gamma_bar <= 0 or self.g_cap < 0 or self.gain <= 0:
            raise ValueError("invalid controller parameters")
        if self.denominator_epsilon < 0:
            raise ValueError("denominator_epsilon must be nonnegative")
        if not self.lower < self.upper:
            raise ValueError("invalid legal image range")


def controller_weight(
    sigma_schedule: float,
    *,
    gamma_bar: float,
    g_cap: float,
    gain: float,
    denominator_epsilon: float = EPSILON,
) -> float:
    """Return the shared prior coefficient for one reverse level."""

    values = (sigma_schedule, gamma_bar, g_cap, gain, denominator_epsilon)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("controller inputs must be finite")
    if sigma_schedule <= 0 or gamma_bar <= 0 or g_cap < 0 or gain <= 0:
        raise ValueError("invalid controller inputs")
    if denominator_epsilon < 0:
        raise ValueError("denominator_epsilon must be nonnegative")
    normalized = min(
        gamma_bar / (sigma_schedule**2 + denominator_epsilon),
        g_cap,
    )
    return float(gain * normalized)


@dataclass(frozen=True)
class TripletPriorStep:
    """Outputs of one frozen 2-D-prior and axial-refiner evaluation."""

    timestep: int
    sigma_schedule: Tensor
    sigma_ddpm: Tensor
    epsilon_2d: Tensor
    tweedie_2d: Tensor
    tweedie_normalized: Tensor
    center_epsilon_residual: Tensor
    anchors: Tensor
    replicated_center_identity: bool


def _require_frozen_eval(module: Any) -> None:
    if not isinstance(module, torch.nn.Module):
        return
    if module.training:
        raise ValueError("the axial refiner bridge must be in evaluation mode")
    if any(parameter.requires_grad for parameter in module.parameters()):
        raise ValueError("the axial refiner bridge must be frozen")


@torch.no_grad()
def triplet_prior_step(
    states: Tensor,
    *,
    sigma_schedule: Tensor | float,
    timestep: int,
    alpha_bar_t: Tensor | float,
    base_forward: BaseForward,
    epsilon_bridge: EpsilonBridge | None,
    lower: float,
    upper: float,
    require_replicated_identity: bool = True,
) -> TripletPriorStep:
    """Form three 2-D anchors and refine only the center anchor."""

    batch, height, width = _require_triplet("states", states)
    if isinstance(timestep, bool) or not isinstance(timestep, int):
        raise TypeError("timestep must be a Python integer")
    if timestep < 0 or timestep > 999:
        raise ValueError("timestep must lie in [0, 999]")
    if not math.isfinite(lower) or not math.isfinite(upper) or not lower < upper:
        raise ValueError("invalid legal image range")

    sigma = torch.as_tensor(
        sigma_schedule,
        device=states.device,
        dtype=states.dtype,
    )
    alpha = torch.as_tensor(alpha_bar_t, device=states.device, dtype=states.dtype)
    if sigma.ndim != 0 or alpha.ndim != 0:
        raise ValueError("sigma_schedule and alpha_bar_t must be scalars")
    if not bool(torch.isfinite(sigma)) or float(sigma) <= 0:
        raise ValueError("sigma_schedule must be finite and positive")
    if not bool(torch.isfinite(alpha)) or not 0.0 < float(alpha) <= 1.0:
        raise ValueError("alpha_bar_t must lie in (0, 1]")

    flat = states.reshape(batch * PHYSICAL_SLICES, 1, height, width)
    x_vp = torch.sqrt(alpha) * flat
    flat_timestep = torch.full(
        (batch * PHYSICAL_SLICES,),
        timestep,
        device=states.device,
        dtype=torch.long,
    )
    epsilon_flat = base_forward(x_vp, flat_timestep)
    if tuple(epsilon_flat.shape) != tuple(flat.shape):
        raise ValueError("base_forward must preserve the flattened triplet shape")
    _require_finite("2-D epsilon prediction", epsilon_flat)
    epsilon_2d = epsilon_flat.reshape_as(states)

    sigma_ddpm = torch.sqrt((1.0 - alpha) / alpha)
    if float(sigma_ddpm) <= 0.0:
        raise ValueError("the selected DDPM timestep must have positive noise")
    tweedie_2d = states - sigma_ddpm * epsilon_2d
    tweedie_normalized = normalize_x0(
        tweedie_2d,
        lower=lower,
        upper=upper,
    )

    batch_timestep = torch.full(
        (batch,),
        timestep,
        device=states.device,
        dtype=torch.long,
    )
    if epsilon_bridge is None:
        center_residual = torch.zeros_like(epsilon_2d[:, CENTER_INDEX])
        replicated_identity = True
    else:
        _require_frozen_eval(epsilon_bridge)
        center_residual = epsilon_bridge(
            tweedie_normalized,
            epsilon_2d,
            batch_timestep,
        )
        if center_residual.shape != epsilon_2d[:, CENTER_INDEX].shape:
            raise ValueError("epsilon bridge returned the wrong center shape")
        _require_finite("center epsilon residual", center_residual)

        replicated_tweedie = tweedie_normalized[:, CENTER_INDEX : CENTER_INDEX + 1]
        replicated_tweedie = replicated_tweedie.expand(
            -1,
            PHYSICAL_SLICES,
            -1,
            -1,
            -1,
        )
        replicated_epsilon = epsilon_2d[:, CENTER_INDEX : CENTER_INDEX + 1]
        replicated_epsilon = replicated_epsilon.expand(
            -1,
            PHYSICAL_SLICES,
            -1,
            -1,
            -1,
        )
        replicated_residual = epsilon_bridge(
            replicated_tweedie,
            replicated_epsilon,
            batch_timestep,
        )
        replicated_identity = bool(
            torch.equal(replicated_residual, torch.zeros_like(replicated_residual))
        )
        if require_replicated_identity and not replicated_identity:
            raise RuntimeError("refiner violates replicated-center exact identity")

    anchors = states - sigma * epsilon_2d
    anchors = anchors.clone()
    anchors[:, CENTER_INDEX] = (
        states[:, CENTER_INDEX]
        - sigma * (epsilon_2d[:, CENTER_INDEX] + center_residual)
    )
    anchors = anchors.clamp(lower, upper)
    _require_finite("prior anchors", anchors)
    return TripletPriorStep(
        timestep=timestep,
        sigma_schedule=sigma.detach().clone(),
        sigma_ddpm=sigma_ddpm.detach().clone(),
        epsilon_2d=epsilon_2d,
        tweedie_2d=tweedie_2d,
        tweedie_normalized=tweedie_normalized,
        center_epsilon_residual=center_residual,
        anchors=anchors,
        replicated_center_identity=replicated_identity,
    )


class ObservationUpdate(Protocol):
    """Per-slice effective-measurement update."""

    def __call__(
        self,
        slice_index: int,
        observation: SliceObservation,
        prior: Tensor,
        step: int,
    ) -> tuple[Tensor, dict[str, Any]]: ...


class FixedObservationUpdate:
    """Return each simulated acquisition's own measurement unchanged."""

    def __call__(
        self,
        slice_index: int,
        observation: SliceObservation,
        prior: Tensor,
        step: int,
    ) -> tuple[Tensor, dict[str, Any]]:
        del prior
        if slice_index < 0 or slice_index >= PHYSICAL_SLICES or step < 0:
            raise ValueError("invalid physical slice or reverse step")
        return observation.measurement, {
            "slice_index": slice_index,
            "label": observation.label,
            "measurement_unchanged": True,
        }


@dataclass(frozen=True)
class RankOneCalibration:
    """Detector direction and coefficient bounds for one physical slice."""

    direction: Tensor
    lower: float
    upper: float
    label: str


class BoundedRankOneObservationUpdate:
    """Maintain one independent bounded detector coefficient per slice."""

    def __init__(
        self,
        calibrations: Sequence[RankOneCalibration],
        projections: Sequence[Projection],
        *,
        activation_step: int = 15,
    ) -> None:
        if len(calibrations) != PHYSICAL_SLICES or (
            len(projections) != PHYSICAL_SLICES
        ):
            raise ValueError("three calibrations and projections are required")
        if activation_step < 0:
            raise ValueError("activation_step must be nonnegative")
        self.calibrations = tuple(calibrations)
        self.projections = tuple(projections)
        self.activation_step = int(activation_step)
        self.coefficients: list[Tensor] = []
        for index, calibration in enumerate(self.calibrations):
            direction = calibration.direction
            if direction.ndim != 1 or direction.numel() < 1:
                raise ValueError("rank-one direction must have shape [detectors]")
            _require_finite("rank-one direction", direction)
            if abs(float(direction.square().sum()) - 1.0) > 2.0e-5:
                raise ValueError("rank-one direction must have unit norm")
            if not math.isfinite(calibration.lower) or not math.isfinite(
                calibration.upper
            ):
                raise ValueError("rank-one coefficient bounds must be finite")
            if calibration.lower > calibration.upper:
                raise ValueError("rank-one coefficient bounds are reversed")
            if calibration.label != SLICE_LABELS[index]:
                raise ValueError("rank-one calibration order is incorrect")
            self.coefficients.append(direction.new_zeros(()))

    @torch.no_grad()
    def __call__(
        self,
        slice_index: int,
        observation: SliceObservation,
        prior: Tensor,
        step: int,
    ) -> tuple[Tensor, dict[str, Any]]:
        calibration = self.calibrations[slice_index]
        if observation.label != calibration.label:
            raise ValueError("observation and calibration labels differ")
        measurement = observation.measurement
        if measurement.ndim != 3 or measurement.shape[0] != 1:
            raise ValueError("measured sinogram must have shape [1, angles, detectors]")
        if measurement.shape[-1] != calibration.direction.numel():
            raise ValueError("measurement and detector direction sizes differ")

        updated = False
        if step >= self.activation_step:
            projected = self.projections[slice_index](prior)
            if projected.shape != measurement.shape:
                raise ValueError("projection and measurement shapes differ")
            _require_finite("current projection", projected)
            detector_residual = (measurement - projected)[0].mean(dim=0)
            coefficient = torch.dot(calibration.direction, detector_residual)
            coefficient = torch.clamp(
                coefficient,
                min=calibration.lower,
                max=calibration.upper,
            )
            self.coefficients[slice_index] = coefficient
            updated = True

        coefficient = self.coefficients[slice_index]
        profile = coefficient * calibration.direction
        effective = measurement - profile[None, None, :]
        return effective, {
            "slice_index": slice_index,
            "label": calibration.label,
            "updated": updated,
            "coefficient": float(coefficient),
            "own_measurement_only": True,
        }


@dataclass(frozen=True)
class OwnSliceDCResult:
    """Three updated images and diagnostics from the separate DC solves."""

    images: Tensor
    effective_measurements: tuple[Tensor, Tensor, Tensor]
    diagnostics: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
    solver_call_count: int


@torch.no_grad()
def apply_own_slice_dc(
    priors: Tensor,
    observations: tuple[SliceObservation, SliceObservation, SliceObservation],
    *,
    step: int,
    gamma: float,
    cg_iterations: int,
    observation_update: ObservationUpdate,
    dc_solvers: tuple[DCSolver, DCSolver, DCSolver],
) -> OwnSliceDCResult:
    """Run three explicit single-slice solves with three own measurements."""

    batch, _, _ = _require_triplet("priors", priors)
    if len(observations) != PHYSICAL_SLICES or len(dc_solvers) != PHYSICAL_SLICES:
        raise ValueError("exactly three observations and solvers are required")
    if step < 0 or cg_iterations < 1 or not math.isfinite(gamma) or gamma < 0:
        raise ValueError("invalid DC controls")

    outputs: list[Tensor] = []
    effective_measurements: list[Tensor] = []
    diagnostics: list[dict[str, Any]] = []
    for slice_index, label in enumerate(SLICE_LABELS):
        observation = observations[slice_index]
        if observation.label != label:
            raise ValueError("physical-slice observation order changed")
        prior = priors[:, slice_index]
        effective, diagnostic = observation_update(
            slice_index,
            observation,
            prior,
            step,
        )
        _require_finite("effective measurement", effective)
        output = dc_solvers[slice_index](
            prior,
            effective,
            float(gamma),
            int(cg_iterations),
        )
        if output.shape != prior.shape or output.shape[0] != batch:
            raise ValueError("single-slice DC solver changed the image shape")
        _require_finite("single-slice DC output", output)
        outputs.append(output)
        effective_measurements.append(effective)
        diagnostics.append(
            {
                **diagnostic,
                "whole_triplet_received_by_solver": False,
            }
        )

    return OwnSliceDCResult(
        images=torch.stack(outputs, dim=1),
        effective_measurements=tuple(effective_measurements),  # type: ignore[arg-type]
        diagnostics=tuple(diagnostics),  # type: ignore[arg-type]
        solver_call_count=PHYSICAL_SLICES,
    )


@torch.no_grad()
def torch_proximal_cg(
    prior: Tensor,
    measurement: Tensor,
    gamma: float,
    iterations: int,
    *,
    forward: Projection,
    adjoint: Projection,
    weights: Tensor | None = None,
) -> Tensor:
    """Solve ``(A'WA + gamma I)x = A'Wy + gamma D`` by CG."""

    if prior.ndim != 4 or prior.shape[1] != 1:
        raise ValueError("prior must have shape [K,1,H,W]")
    if measurement.shape[0] != prior.shape[0]:
        raise ValueError("measurement and prior batch sizes differ")
    if iterations < 1 or not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("iterations and gamma must be positive")
    _require_finite("prior", prior)
    _require_finite("measurement", measurement)
    if weights is not None:
        _require_finite("weights", weights)
        if bool((weights < 0).any()):
            raise ValueError("weights must be nonnegative")

    def apply_weights(value: Tensor) -> Tensor:
        return value if weights is None else weights * value

    def normal(value: Tensor) -> Tensor:
        return adjoint(apply_weights(forward(value))) + gamma * value

    estimate = prior.clone()
    rhs = adjoint(apply_weights(measurement)) + gamma * prior
    residual = rhs - normal(estimate)
    direction = residual.clone()
    reduce_dims = tuple(range(1, residual.ndim))
    residual_square = (residual * residual).sum(dim=reduce_dims)
    broadcast = (prior.shape[0],) + (1,) * (prior.ndim - 1)

    for _ in range(iterations):
        applied = normal(direction)
        denominator = (direction * applied).sum(dim=reduce_dims)
        alpha = residual_square / (denominator + EPSILON)
        estimate = estimate + alpha.view(broadcast) * direction
        residual = residual - alpha.view(broadcast) * applied
        next_square = (residual * residual).sum(dim=reduce_dims)
        beta = next_square / (residual_square + EPSILON)
        direction = residual + beta.view(broadcast) * direction
        residual_square = next_square
    return estimate


class IndependentTripletNoise:
    """Three persistent random streams, one for each physical state."""

    def __init__(self, *, device: torch.device | str, seeds: Sequence[int]) -> None:
        if len(seeds) != PHYSICAL_SLICES:
            raise ValueError("exactly three seeds are required")
        if len({int(seed) for seed in seeds}) != PHYSICAL_SLICES:
            raise ValueError("the three physical-state seeds must be distinct")
        resolved = torch.device(device)
        if resolved.type == "cuda" and resolved.index is None:
            resolved = torch.device("cuda", torch.cuda.current_device())
        self.device = resolved
        self.seeds = tuple(int(seed) for seed in seeds)
        self.generators = tuple(
            torch.Generator(device=resolved).manual_seed(seed) for seed in self.seeds
        )
        self.draw_count = 0

    def draw(self, template: Tensor) -> Tensor:
        _require_triplet("noise template", template)
        if template.device != self.device:
            raise ValueError("noise template and generators use different devices")
        shape = tuple(template[:, 0].shape)
        values = [
            torch.randn(
                shape,
                device=self.device,
                dtype=template.dtype,
                generator=generator,
            )
            for generator in self.generators
        ]
        self.draw_count += 1
        return torch.stack(values, dim=1)


def _validate_schedule(sigmas: Tensor, expected_levels: int) -> None:
    if sigmas.ndim != 1 or len(sigmas) != expected_levels + 1:
        raise ValueError("sigma_steps must contain num_levels active values and zero")
    _require_finite("sigma_steps", sigmas)
    if not bool((sigmas[:-1] > 0).all()) or float(sigmas[-1]) != 0.0:
        raise ValueError("sigma_steps must contain positive levels followed by zero")
    if not bool((sigmas[1:] <= sigmas[:-1]).all()):
        raise ValueError("sigma_steps must be nonincreasing")


@dataclass(frozen=True)
class TripletSamplerResult:
    """Center reconstruction, terminal triplet, and numerical trace."""

    center: Tensor
    terminal_triplet: Tensor
    gamma_schedule: tuple[float, ...]
    timesteps: tuple[int, ...]
    dc_call_count: int
    noise_draw_count: int
    replicated_center_identity: bool


@torch.no_grad()
def sample_triplet(
    *,
    initial_template: Tensor,
    observations: tuple[SliceObservation, SliceObservation, SliceObservation],
    sigma_steps: Tensor,
    alpha_cumprod: Tensor,
    sigma_ddpm: Tensor,
    config: TripletSamplerConfig,
    base_forward: BaseForward,
    epsilon_bridge: EpsilonBridge,
    seeds: Sequence[int],
    dc_solvers: tuple[DCSolver, DCSolver, DCSolver],
    observation_update: ObservationUpdate | None = None,
) -> TripletSamplerResult:
    """Run the interior trajectory and return only its center as reconstruction."""

    config.validate()
    batch, _, _ = _require_triplet("initial_template", initial_template)
    if batch != config.ensemble_size:
        raise ValueError("initial template and configured trajectory count differ")
    _validate_schedule(sigma_steps, config.num_levels)
    if alpha_cumprod.ndim != 1 or sigma_ddpm.ndim != 1:
        raise ValueError("DDPM schedules must be one-dimensional")
    if len(alpha_cumprod) != 1000 or len(sigma_ddpm) != 1000:
        raise ValueError("the base DDPM schedules must contain 1000 steps")
    _require_finite("alpha_cumprod", alpha_cumprod)
    _require_finite("sigma_ddpm", sigma_ddpm)
    _require_frozen_eval(epsilon_bridge)

    update = observation_update or FixedObservationUpdate()
    streams = IndependentTripletNoise(device=initial_template.device, seeds=seeds)
    state = sigma_steps[0] * streams.draw(initial_template)
    gammas: list[float] = []
    timesteps: list[int] = []
    dc_calls = 0
    replicated_identity = True

    for step in range(config.num_levels):
        sigma = sigma_steps[step]
        timestep = int((sigma_ddpm - sigma).abs().argmin())
        prior = triplet_prior_step(
            state,
            sigma_schedule=sigma,
            timestep=timestep,
            alpha_bar_t=alpha_cumprod[timestep],
            base_forward=base_forward,
            epsilon_bridge=epsilon_bridge,
            lower=config.lower,
            upper=config.upper,
        )
        gamma = controller_weight(
            float(sigma),
            gamma_bar=config.gamma_bar,
            g_cap=config.g_cap,
            gain=config.gain,
            denominator_epsilon=config.denominator_epsilon,
        )
        dc = apply_own_slice_dc(
            prior.anchors,
            observations,
            step=step,
            gamma=gamma,
            cg_iterations=config.cg_iterations,
            observation_update=update,
            dc_solvers=dc_solvers,
        )
        state = dc.images + sigma_steps[step + 1] * streams.draw(initial_template)
        dc_calls += dc.solver_call_count
        gammas.append(gamma)
        timesteps.append(timestep)
        replicated_identity = (
            replicated_identity and prior.replicated_center_identity
        )

    expected_calls = PHYSICAL_SLICES * config.num_levels
    if dc_calls != expected_calls or streams.draw_count != config.num_levels + 1:
        raise RuntimeError("triplet solver or random-stream call count changed")
    return TripletSamplerResult(
        center=state[:, CENTER_INDEX],
        terminal_triplet=state,
        gamma_schedule=tuple(gammas),
        timesteps=tuple(timesteps),
        dc_call_count=dc_calls,
        noise_draw_count=streams.draw_count,
        replicated_center_identity=replicated_identity,
    )


@dataclass(frozen=True)
class CenterOnlySamplerResult:
    """Result of the adapter-free boundary path."""

    center: Tensor
    gamma_schedule: tuple[float, ...]
    dc_call_count: int
    noise_draw_count: int


@torch.no_grad()
def sample_center_only(
    *,
    initial_template: Tensor,
    observation: SliceObservation,
    sigma_steps: Tensor,
    alpha_cumprod: Tensor,
    sigma_ddpm: Tensor,
    config: TripletSamplerConfig,
    base_forward: BaseForward,
    seed: int,
    dc_solver: DCSolver,
) -> CenterOnlySamplerResult:
    """Run the true single-state, adapter-free path used at volume boundaries."""

    config.validate()
    if initial_template.ndim != 4 or tuple(initial_template.shape[:2]) != (1, 1):
        raise ValueError("initial_template must have shape [1,1,H,W]")
    if observation.label != "z":
        raise ValueError("the boundary path accepts only the center observation")
    _validate_schedule(sigma_steps, config.num_levels)
    if len(alpha_cumprod) != 1000 or len(sigma_ddpm) != 1000:
        raise ValueError("the base DDPM schedules must contain 1000 steps")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    generator = torch.Generator(device=initial_template.device).manual_seed(seed)

    def draw() -> Tensor:
        return torch.randn(
            tuple(initial_template.shape),
            device=initial_template.device,
            dtype=initial_template.dtype,
            generator=generator,
        )

    draw_count = 1
    state = sigma_steps[0] * draw()
    gammas: list[float] = []
    for step in range(config.num_levels):
        sigma = sigma_steps[step]
        timestep = int((sigma_ddpm - sigma).abs().argmin())
        alpha = alpha_cumprod[timestep]
        timestep_tensor = torch.full(
            (1,),
            timestep,
            device=state.device,
            dtype=torch.long,
        )
        epsilon = base_forward(torch.sqrt(alpha) * state, timestep_tensor)
        if epsilon.shape != state.shape:
            raise ValueError("base_forward changed the center-state shape")
        anchor = (state - sigma * epsilon).clamp(config.lower, config.upper)
        gamma = controller_weight(
            float(sigma),
            gamma_bar=config.gamma_bar,
            g_cap=config.g_cap,
            gain=config.gain,
            denominator_epsilon=config.denominator_epsilon,
        )
        state = dc_solver(
            anchor,
            observation.measurement,
            gamma,
            config.cg_iterations,
        )
        if state.shape != anchor.shape:
            raise ValueError("boundary DC solver changed the image shape")
        state = state + sigma_steps[step + 1] * draw()
        draw_count += 1
        gammas.append(gamma)

    return CenterOnlySamplerResult(
        center=state,
        gamma_schedule=tuple(gammas),
        dc_call_count=config.num_levels,
        noise_draw_count=draw_count,
    )
