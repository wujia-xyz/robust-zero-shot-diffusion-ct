import unittest

import numpy as np

from robust_ct.proximal import proximal_cg


class ProximalCGTests(unittest.TestCase):
    def test_matches_small_weighted_exact_solution(self) -> None:
        matrix = np.array([[2.0, 0.5], [0.0, 1.5], [1.0, -0.25]])
        measurement = np.array([1.0, -2.0, 0.5])
        weights = np.array([1.0, 0.25, 2.0])
        prior = np.array([0.2, -0.4])
        gamma = 0.7

        result = proximal_cg(
            lambda image: matrix @ image,
            lambda projection: matrix.T @ projection,
            measurement,
            prior,
            gamma,
            iterations=8,
            weights=weights,
            rtol=1.0e-13,
        )
        normal = matrix.T @ np.diag(weights) @ matrix + gamma * np.eye(2)
        rhs = matrix.T @ (weights * measurement) + gamma * prior
        expected = np.linalg.solve(normal, rhs)
        np.testing.assert_allclose(result.solution, expected, rtol=1e-11, atol=1e-11)
        self.assertLessEqual(result.iterations, 2)

    def test_default_warm_start_is_prior_center(self) -> None:
        prior = np.array([1.5, -0.5])
        result = proximal_cg(
            lambda image: image,
            lambda projection: projection,
            measurement=prior,
            prior_center=prior,
            prior_weight=2.0,
            iterations=6,
        )
        np.testing.assert_array_equal(result.solution, prior)
        self.assertEqual(result.iterations, 0)
        self.assertTrue(result.converged)


if __name__ == "__main__":
    unittest.main()
