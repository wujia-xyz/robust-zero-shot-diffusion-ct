"""Acquisition-conditioned axial prior refinement for zero-shot CT."""

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
    "AXIAL_AVAILABLE",
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


# The scalar controller and NumPy numerical core remain usable without
# PyTorch.  Axial and triplet symbols are exported when the optional dependency
# is installed.
AXIAL_AVAILABLE = False
try:
    from .axial import (
        EXPECTED_PARAMETER_COUNT,
        AxialCenterX0Adapter,
        AxialCenterX0Refiner,
        X0ToEpsilonResidualBridge,
        affine_normalize_x0,
        candidate_center_x0_normalized,
        corrected_epsilon_from_delta_x0,
        delta_x0_norm_to_physical,
        normalize_x0,
        normalized_x0_to_physical,
        parameter_count,
        refined_center_anchor,
    )
    from .triplet import (
        BoundedRankOneObservationUpdate,
        CenterOnlySamplerResult,
        FixedObservationUpdate,
        OwnSliceDCResult,
        RankOneCalibration,
        SliceObservation,
        TripletPriorStep,
        TripletSamplerConfig,
        TripletSamplerResult,
        apply_own_slice_dc,
        controller_weight,
        sample_center_only,
        sample_triplet,
        torch_proximal_cg,
        triplet_prior_step,
    )
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
else:
    AXIAL_AVAILABLE = True
    __all__ += [
        "EXPECTED_PARAMETER_COUNT",
        "AxialCenterX0Adapter",
        "AxialCenterX0Refiner",
        "BoundedRankOneObservationUpdate",
        "CenterOnlySamplerResult",
        "FixedObservationUpdate",
        "OwnSliceDCResult",
        "RankOneCalibration",
        "SliceObservation",
        "TripletPriorStep",
        "TripletSamplerConfig",
        "TripletSamplerResult",
        "X0ToEpsilonResidualBridge",
        "affine_normalize_x0",
        "apply_own_slice_dc",
        "candidate_center_x0_normalized",
        "controller_weight",
        "corrected_epsilon_from_delta_x0",
        "delta_x0_norm_to_physical",
        "normalize_x0",
        "normalized_x0_to_physical",
        "parameter_count",
        "refined_center_anchor",
        "sample_center_only",
        "sample_triplet",
        "torch_proximal_cg",
        "triplet_prior_step",
    ]

__version__ = "0.2.1"
