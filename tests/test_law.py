import unittest

from robust_ct.law import (
    AcquisitionStatistics,
    OperatingState,
    acquisition_conditioned_cap,
    calibrate_sampler_scale,
    compute_normalized_cap,
    realize_terminal_operating_point,
    relative_cap_ratio,
)


class CapLawTests(unittest.TestCase):
    def test_cap_formula_and_calibration_round_trip(self) -> None:
        statistics = AcquisitionStatistics(
            noise_variance=0.4,
            prior_error_rms=0.2,
            gain=5.0,
        )
        sampler_scale = 0.75
        cap = acquisition_conditioned_cap(statistics, sampler_scale)
        self.assertAlmostEqual(cap, 0.75 / 5.0 * (1.0 + 0.4 / 0.04))
        self.assertAlmostEqual(
            calibrate_sampler_scale(cap, statistics),
            sampler_scale,
        )
        self.assertAlmostEqual(
            compute_normalized_cap(
                noise_variance=statistics.noise_variance,
                prior_error=statistics.prior_error_rms,
                gain=statistics.gain,
                calibration_factor=sampler_scale,
            ),
            cap,
        )

    def test_relative_ratio_is_independent_of_sampler_scale(self) -> None:
        first = AcquisitionStatistics(0.03, 0.1, 20.0)
        second = AcquisitionStatistics(0.08, 0.2, 35.0)
        expected = relative_cap_ratio(first, second)
        for scale in (0.1, 0.7, 9.0):
            direct = (
                acquisition_conditioned_cap(first, scale)
                / acquisition_conditioned_cap(second, scale)
            )
            self.assertAlmostEqual(direct, expected)

    def test_terminal_state_selection(self) -> None:
        active = realize_terminal_operating_point(
            cap=4.0,
            gamma_bar=0.1,
            sigma_min=0.1,
            denominator_epsilon=0.0,
        )
        self.assertEqual(active.state, OperatingState.CAP_ACTIVE)
        self.assertEqual(active.realized, 4.0)
        self.assertAlmostEqual(active.endpoint, 10.0)

        limited = realize_terminal_operating_point(
            cap=12.0,
            gamma_bar=0.1,
            sigma_min=0.1,
            denominator_epsilon=0.0,
        )
        self.assertEqual(limited.state, OperatingState.ENDPOINT_LIMITED)
        self.assertAlmostEqual(limited.realized, 10.0)


if __name__ == "__main__":
    unittest.main()
