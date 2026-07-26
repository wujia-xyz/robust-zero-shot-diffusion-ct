"""Reference implementation of robust zero-shot diffusion CT conditioning."""

from .law import (
    AcquisitionStatistics,
    OperatingState,
    TerminalOperatingPoint,
    acquisition_conditioned_cap,
    calibrate_sampler_scale,
    compute_normalized_cap,
    endpoint_coordinate,
    realize_terminal_operating_point,
    relative_cap_ratio,
)
from .proximal import CGResult, proximal_cg
from .reliability import DetectorReliability, detector_column_reliability
from .sampler import SamplerResult, reconstruct
from .schedule import (
    normalized_anchor_trajectory,
    power_noise_schedule,
    prior_anchor_trajectory,
    select_two_regime_gamma_bar,
)

__all__ = [
    "AcquisitionStatistics",
    "CGResult",
    "DetectorReliability",
    "OperatingState",
    "SamplerResult",
    "TerminalOperatingPoint",
    "acquisition_conditioned_cap",
    "calibrate_sampler_scale",
    "compute_normalized_cap",
    "detector_column_reliability",
    "endpoint_coordinate",
    "normalized_anchor_trajectory",
    "power_noise_schedule",
    "prior_anchor_trajectory",
    "proximal_cg",
    "realize_terminal_operating_point",
    "reconstruct",
    "relative_cap_ratio",
    "select_two_regime_gamma_bar",
]
