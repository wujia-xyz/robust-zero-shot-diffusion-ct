"""Compact axial x0 refiner and the two-noise-scale inference bridge.

The refiner consumes clipped, normalized Tweedie estimates and noise-prediction
features from the physical triplet ``(z-1, z, z+1)``.  It predicts a residual
for the center clean estimate only.  The bridge converts that residual to the
epsilon correction expected by a Karras-coordinate CT sampler.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


EXPECTED_PARAMETER_COUNT = 5_984
PHYSICAL_SLICES = 3
CENTER_INDEX = 1


def fixed_sinusoidal_embedding(
    timestep: torch.Tensor,
    *,
    dimension: int = 32,
    maximum_period: float = 10_000.0,
) -> torch.Tensor:
    """Return the deterministic sinusoidal embedding used by the refiner."""

    if dimension < 2 or dimension % 2:
        raise ValueError("embedding dimension must be a positive even integer")
    value = torch.as_tensor(timestep)
    if value.ndim == 0:
        value = value[None]
    if value.ndim != 1:
        raise ValueError(f"timestep must have shape [B], got {tuple(value.shape)}")
    if value.dtype == torch.bool or not bool(torch.isfinite(value).all()):
        raise ValueError("timestep must contain finite integer-like values")
    if not bool((value == value.round()).all()) or not bool(
        ((value >= 0) & (value <= 999)).all()
    ):
        raise ValueError("timestep must contain integers in [0, 999]")

    half = dimension // 2
    exponent = (
        -math.log(maximum_period)
        * torch.arange(half, device=value.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    phase = value.to(torch.float32)[:, None] * torch.exp(exponent)[None, :]
    return torch.cat((torch.sin(phase), torch.cos(phase)), dim=1)


def _validate_range(lower: float, upper: float) -> None:
    if not math.isfinite(lower) or not math.isfinite(upper) or not lower < upper:
        raise ValueError("invalid legal image range")


def affine_normalize_x0(
    value: torch.Tensor,
    *,
    lower: float,
    upper: float,
) -> torch.Tensor:
    """Map physical x0 values to normalized coordinates without clipping."""

    _validate_range(lower, upper)
    if not bool(torch.isfinite(value).all()):
        raise ValueError("x0 contains non-finite values")
    return 2.0 * (value - lower) / (upper - lower) - 1.0


def normalize_x0(
    value: torch.Tensor,
    *,
    lower: float,
    upper: float,
) -> torch.Tensor:
    """Clip a physical image to its legal range and map it to ``[-1, 1]``."""

    _validate_range(lower, upper)
    if not bool(torch.isfinite(value).all()):
        raise ValueError("x0 contains non-finite values")
    return affine_normalize_x0(
        torch.clamp(value, min=lower, max=upper),
        lower=lower,
        upper=upper,
    )


def normalized_x0_to_physical(
    value_normalized: torch.Tensor,
    *,
    lower: float,
    upper: float,
) -> torch.Tensor:
    """Map normalized x0 coordinates back to physical image units."""

    _validate_range(lower, upper)
    if not bool(torch.isfinite(value_normalized).all()):
        raise ValueError("normalized x0 contains non-finite values")
    return lower + 0.5 * (upper - lower) * (value_normalized + 1.0)


def delta_x0_norm_to_physical(
    delta_x0_normalized: torch.Tensor,
    *,
    lower: float,
    upper: float,
) -> torch.Tensor:
    """Convert a normalized residual without adding the affine offset."""

    _validate_range(lower, upper)
    if not bool(torch.isfinite(delta_x0_normalized).all()):
        raise ValueError("normalized x0 residual contains non-finite values")
    return 0.5 * (upper - lower) * delta_x0_normalized


def _positive_scalar_like(
    value: torch.Tensor | float,
    reference: torch.Tensor,
    label: str,
) -> torch.Tensor:
    result = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if result.ndim != 0 or not bool(torch.isfinite(result)) or float(result) <= 0.0:
        raise ValueError(f"{label} must be a finite positive scalar")
    return result


def epsilon_correction_from_delta_x0(
    delta_x0: torch.Tensor,
    *,
    sigma_ddpm: torch.Tensor | float,
) -> torch.Tensor:
    """Return the equivalent epsilon correction ``-delta_x0/sigma_ddpm``."""

    if not bool(torch.isfinite(delta_x0).all()):
        raise ValueError("delta_x0 contains non-finite values")
    sigma = _positive_scalar_like(sigma_ddpm, delta_x0, "sigma_ddpm")
    return -delta_x0 / sigma


def corrected_epsilon_from_delta_x0(
    epsilon_base: torch.Tensor,
    delta_x0: torch.Tensor,
    *,
    sigma_ddpm: torch.Tensor | float,
) -> torch.Tensor:
    """Add the x0-derived correction to a center epsilon prediction."""

    if epsilon_base.shape != delta_x0.shape:
        raise ValueError("epsilon_base and delta_x0 shapes differ")
    return epsilon_base + epsilon_correction_from_delta_x0(
        delta_x0,
        sigma_ddpm=sigma_ddpm,
    )


def refined_center_anchor(
    center_state: torch.Tensor,
    epsilon_base: torch.Tensor,
    delta_x0: torch.Tensor,
    *,
    sigma_schedule: torch.Tensor | float,
    sigma_ddpm: torch.Tensor | float,
) -> torch.Tensor:
    """Construct the refined center anchor using the two distinct sigmas."""

    if not (center_state.shape == epsilon_base.shape == delta_x0.shape):
        raise ValueError("center-state, epsilon, and residual shapes differ")
    schedule_sigma = _positive_scalar_like(
        sigma_schedule,
        delta_x0,
        "sigma_schedule",
    )
    ddpm_sigma = _positive_scalar_like(sigma_ddpm, delta_x0, "sigma_ddpm")
    return (
        center_state
        - schedule_sigma * epsilon_base
        + (schedule_sigma / ddpm_sigma) * delta_x0
    )


class AxialCenterX0Refiner(nn.Module):
    """Predict a center normalized-x0 residual from a three-slice context."""

    def __init__(self, width: int = 32, timestep_dimension: int = 32) -> None:
        super().__init__()
        if width != 32 or timestep_dimension != 32:
            raise ValueError("the released architecture uses width=32 and time_dim=32")
        self.width = width
        self.timestep_dimension = timestep_dimension
        self.stem = nn.Conv2d(2, width, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, width)
        self.time1 = nn.Linear(timestep_dimension, width)
        self.time2 = nn.Linear(width, width)
        self.mix = nn.Conv2d(3 * width, width, kernel_size=1)
        self.norm2 = nn.GroupNorm(8, width)
        self.zero_out = nn.Conv2d(width, 1, kernel_size=1, bias=False)
        nn.init.zeros_(self.zero_out.weight)
        if parameter_count(self) != EXPECTED_PARAMETER_COUNT:
            raise RuntimeError("axial-refiner parameter count drifted")

    def forward(
        self,
        tweedie_normalized: torch.Tensor,
        epsilon_2d: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        if tweedie_normalized.shape != epsilon_2d.shape:
            raise ValueError("Tweedie and epsilon shapes differ")
        if epsilon_2d.ndim != 5 or tuple(epsilon_2d.shape[1:3]) != (3, 1):
            raise ValueError(
                "expected [B,3,1,H,W] Tweedie/epsilon tensors, got "
                f"{tuple(epsilon_2d.shape)}"
            )
        if not bool(torch.isfinite(tweedie_normalized).all()) or not bool(
            torch.isfinite(epsilon_2d).all()
        ):
            raise ValueError("non-finite refiner input")

        batch, _, _, height, width = epsilon_2d.shape
        timestep_value = torch.as_tensor(timestep, device=epsilon_2d.device)
        if timestep_value.ndim == 0:
            timestep_value = timestep_value.expand(batch)
        if tuple(timestep_value.shape) != (batch,):
            raise ValueError(f"timestep must have shape [{batch}]")

        features = torch.cat((tweedie_normalized, epsilon_2d), dim=2)
        features = features.reshape(batch * PHYSICAL_SLICES, 2, height, width)
        hidden = self.stem(features)
        time = fixed_sinusoidal_embedding(
            timestep_value,
            dimension=self.timestep_dimension,
        ).to(hidden.dtype)
        time = self.time2(F.silu(self.time1(time)))
        time = time[:, None, :].expand(batch, PHYSICAL_SLICES, self.width)
        hidden = hidden + time.reshape(batch * PHYSICAL_SLICES, self.width, 1, 1)
        hidden = F.silu(self.norm1(hidden))
        hidden = hidden.reshape(batch, PHYSICAL_SLICES, self.width, height, width)

        negative, center, positive = hidden.unbind(dim=1)
        contextual = torch.cat(
            (
                center,
                0.5 * (negative + positive),
                torch.abs(negative - positive),
            ),
            dim=1,
        )
        replicated_center = torch.cat(
            (center, center, torch.zeros_like(center)),
            dim=1,
        )
        contextual_value = self.zero_out(F.silu(self.norm2(self.mix(contextual))))
        center_value = self.zero_out(
            F.silu(self.norm2(self.mix(replicated_center)))
        )
        return contextual_value - center_value


class X0ToEpsilonResidualBridge(nn.Module):
    """Expose the normalized-x0 refiner through an epsilon-residual interface."""

    def __init__(
        self,
        refiner: nn.Module,
        sigma_ddpm: torch.Tensor,
        *,
        lower: float,
        upper: float,
    ) -> None:
        super().__init__()
        if sigma_ddpm.ndim != 1 or len(sigma_ddpm) != 1000:
            raise ValueError("sigma_ddpm must contain the 1000-step DDPM schedule")
        if not bool(torch.isfinite(sigma_ddpm).all()) or not bool(
            (sigma_ddpm > 0).all()
        ):
            raise ValueError("sigma_ddpm must be finite and positive")
        _validate_range(lower, upper)
        self.refiner = refiner
        self.register_buffer(
            "sigma_ddpm",
            sigma_ddpm.detach().clone().to(torch.float32),
        )
        self.lower = float(lower)
        self.upper = float(upper)

    def forward(
        self,
        tweedie_normalized: torch.Tensor,
        epsilon_2d: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        delta_normalized = self.refiner(
            tweedie_normalized,
            epsilon_2d,
            timestep,
        )
        delta_x0 = delta_x0_norm_to_physical(
            delta_normalized,
            lower=self.lower,
            upper=self.upper,
        )
        index = torch.as_tensor(
            timestep,
            device=delta_x0.device,
            dtype=torch.long,
        )
        if index.ndim == 0:
            index = index[None]
        if tuple(index.shape) != (delta_x0.shape[0],):
            raise ValueError("timestep batch differs from refiner output batch")
        if not bool(((index >= 0) & (index < len(self.sigma_ddpm))).all()):
            raise ValueError("timestep is outside the DDPM schedule")
        sigma = self.sigma_ddpm[index].to(delta_x0.dtype)[:, None, None, None]
        result = -delta_x0 / sigma
        if not bool(torch.isfinite(result).all()):
            raise ValueError("epsilon bridge produced non-finite values")
        return result


def candidate_center_x0_normalized(
    refiner: nn.Module,
    base_tweedie_raw_normalized: torch.Tensor,
    tweedie_feature_normalized: torch.Tensor,
    epsilon_2d: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    """Add the predicted residual to the unclipped normalized center baseline."""

    center_shape = tweedie_feature_normalized[:, CENTER_INDEX].shape
    if base_tweedie_raw_normalized.ndim != 4 or (
        base_tweedie_raw_normalized.shape != center_shape
    ):
        raise ValueError("raw normalized center baseline has the wrong shape")
    if not bool(torch.isfinite(base_tweedie_raw_normalized).all()):
        raise ValueError("raw normalized center baseline is non-finite")
    return base_tweedie_raw_normalized + refiner(
        tweedie_feature_normalized,
        epsilon_2d,
        timestep,
    )


def parameter_count(module: nn.Module) -> int:
    """Return the number of scalar parameters in a module."""

    return int(sum(parameter.numel() for parameter in module.parameters()))


# The historical class name is kept as a source-compatible alias.
AxialCenterX0Adapter = AxialCenterX0Refiner
