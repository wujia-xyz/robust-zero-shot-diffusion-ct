"""Acquisition-conditioned data-consistency cap law.

This module is an independent NumPy implementation of the scalar law described
in the accompanying paper.  It contains no dataset, checkpoint, or
framework-specific code.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


DEFAULT_DENOMINATOR_EPSILON = 1.0e-8


def _finite_nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative; got {value!r}.")
    return value


def _finite_positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive; got {value!r}.")
    return value


@dataclass(frozen=True)
class AcquisitionStatistics:
    r"""Pre-reconstruction statistics used by the cap law.

    Parameters
    ----------
    noise_variance:
        Scalar measurement-noise proxy :math:`\sigma_n^2`.
    prior_error_rms:
        Stored terminal clean-prior RMSE :math:`\delta`.
    gain:
        Positive response scale of the forward operator.
    """

    noise_variance: float
    prior_error_rms: float
    gain: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "noise_variance",
            _finite_nonnegative(self.noise_variance, "noise_variance"),
        )
        object.__setattr__(
            self,
            "prior_error_rms",
            _finite_positive(self.prior_error_rms, "prior_error_rms"),
        )
        object.__setattr__(self, "gain", _finite_positive(self.gain, "gain"))

    @property
    def normalized_uncertainty(self) -> float:
        r"""Return :math:`1+\sigma_n^2/\delta^2`."""

        return 1.0 + self.noise_variance / (self.prior_error_rms**2)


class OperatingState(str, Enum):
    """Terminal state selected by the cap and the annealed backbone."""

    CAP_ACTIVE = "cap-active"
    ENDPOINT_LIMITED = "endpoint-limited"


@dataclass(frozen=True)
class TerminalOperatingPoint:
    """Realized normalized terminal anchor and its operating state."""

    cap: float
    endpoint: float
    realized: float
    state: OperatingState


def acquisition_conditioned_cap(
    statistics: AcquisitionStatistics,
    sampler_scale: float,
) -> float:
    r"""Compute the normalized acquisition-conditioned cap.

    The implemented law is

    .. math::

       g_{\rm cap}=\frac{c}{\mathrm{GAIN}}
       \left(1+\frac{\sigma_n^2}{\delta^2}\right).
    """

    scale = _finite_nonnegative(sampler_scale, "sampler_scale")
    return scale * statistics.normalized_uncertainty / statistics.gain


def compute_normalized_cap(
    noise_variance: float,
    prior_error: float,
    gain: float,
    calibration_factor: float,
) -> float:
    """Convenience wrapper for :func:`acquisition_conditioned_cap`.

    This scalar interface mirrors the notation used in the release README.
    ``prior_error`` is the stored terminal clean-prior RMSE and
    ``calibration_factor`` is the once-calibrated sampler scale ``c``.
    """

    statistics = AcquisitionStatistics(
        noise_variance=noise_variance,
        prior_error_rms=prior_error,
        gain=gain,
    )
    return acquisition_conditioned_cap(statistics, calibration_factor)


def calibrate_sampler_scale(
    reference_cap: float,
    reference_statistics: AcquisitionStatistics,
) -> float:
    """Recover the one-time sampler scale from a reference acquisition."""

    cap = _finite_nonnegative(reference_cap, "reference_cap")
    delta_sq = reference_statistics.prior_error_rms**2
    return (
        cap
        * reference_statistics.gain
        * delta_sq
        / (reference_statistics.noise_variance + delta_sq)
    )


def relative_cap_ratio(
    numerator: AcquisitionStatistics,
    denominator: AcquisitionStatistics,
) -> float:
    """Return ``g_cap(numerator) / g_cap(denominator)`` without using ``c``."""

    return (
        denominator.gain
        / numerator.gain
        * numerator.normalized_uncertainty
        / denominator.normalized_uncertainty
    )


def endpoint_coordinate(
    gamma_bar: float,
    sigma_min: float,
    *,
    denominator_epsilon: float = DEFAULT_DENOMINATOR_EPSILON,
) -> float:
    """Return the normalized uncapped endpoint of the annealed backbone."""

    gamma_bar = _finite_nonnegative(gamma_bar, "gamma_bar")
    sigma_min = _finite_positive(sigma_min, "sigma_min")
    denominator_epsilon = _finite_nonnegative(
        denominator_epsilon, "denominator_epsilon"
    )
    return gamma_bar / (sigma_min**2 + denominator_epsilon)


def realize_terminal_operating_point(
    cap: float,
    gamma_bar: float,
    sigma_min: float,
    *,
    denominator_epsilon: float = DEFAULT_DENOMINATOR_EPSILON,
) -> TerminalOperatingPoint:
    """Combine the prescribed cap with the annealed-backbone endpoint."""

    cap = _finite_nonnegative(cap, "cap")
    endpoint = endpoint_coordinate(
        gamma_bar,
        sigma_min,
        denominator_epsilon=denominator_epsilon,
    )
    if cap < endpoint:
        state = OperatingState.CAP_ACTIVE
        realized = cap
    else:
        state = OperatingState.ENDPOINT_LIMITED
        realized = endpoint
    return TerminalOperatingPoint(
        cap=cap,
        endpoint=endpoint,
        realized=realized,
        state=state,
    )
