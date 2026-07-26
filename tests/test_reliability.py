import unittest

import numpy as np

from robust_ct.reliability import detector_column_reliability


class DetectorReliabilityTests(unittest.TestCase):
    def test_detects_persistent_column_outlier(self) -> None:
        detectors = 41
        profile = np.linspace(-0.2, 0.2, detectors)
        sinogram = np.repeat(profile[None, :], 30, axis=0)
        sinogram[:, 20] += 4.0

        result = detector_column_reliability(sinogram, width=11, threshold=3.0)

        self.assertFalse(result.reliable[20])
        self.assertEqual(result.weights[20], 0.0)
        self.assertTrue(result.reliable[10])
        self.assertEqual(result.weights[10], 1.0)
        self.assertEqual(result.weights.shape, (detectors,))

    def test_rejects_even_filter_width(self) -> None:
        with self.assertRaises(ValueError):
            detector_column_reliability(np.ones((5, 12)), width=4)


if __name__ == "__main__":
    unittest.main()
