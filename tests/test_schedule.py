import unittest

import numpy as np

from robust_ct.schedule import (
    normalized_anchor_trajectory,
    power_noise_schedule,
    select_two_regime_gamma_bar,
)


class ScheduleTests(unittest.TestCase):
    def test_power_schedule_has_requested_endpoints(self) -> None:
        levels = power_noise_schedule(7, sigma_max=12.0, sigma_min=0.01)
        self.assertEqual(levels.shape, (8,))
        self.assertAlmostEqual(levels[0], 12.0)
        self.assertAlmostEqual(levels[-2], 0.01)
        self.assertEqual(levels[-1], 0.0)
        self.assertTrue(np.all(np.diff(levels) < 0.0))

    def test_anchor_schedule_saturates_at_cap(self) -> None:
        anchors = normalized_anchor_trajectory(
            np.array([2.0, 1.0, 0.1]),
            gamma_bar=0.2,
            cap=3.0,
            denominator_epsilon=0.0,
        )
        np.testing.assert_allclose(anchors, [0.05, 0.2, 3.0])

    def test_two_regime_rule(self) -> None:
        self.assertEqual(select_two_regime_gamma_bar(0.0), 0.3)
        self.assertEqual(select_two_regime_gamma_bar(1.0e-12), 0.1)


if __name__ == "__main__":
    unittest.main()
