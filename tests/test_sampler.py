import unittest

import numpy as np

from robust_ct.sampler import reconstruct


class SamplerTests(unittest.TestCase):
    def test_one_step_identity_problem_matches_proximal_solution(self) -> None:
        measurement = np.array([2.0, -1.0])
        gamma = 0.5
        result = reconstruct(
            measurement,
            forward=lambda image: image,
            adjoint=lambda projection: projection,
            denoise=lambda state, sigma: np.zeros_like(state),
            sigma_levels=np.array([1.0, 0.0]),
            gain=1.0,
            gamma_bar=gamma,
            cap=10.0,
            cg_iterations=2,
            chain_count=3,
            seed=9,
            denominator_epsilon=0.0,
        )
        expected = measurement / (1.0 + gamma)
        np.testing.assert_allclose(result.reconstruction, expected)
        np.testing.assert_allclose(result.chains, np.repeat(expected[None, :], 3, axis=0))
        np.testing.assert_allclose(result.prior_anchor_weights, [gamma])

    def test_seeded_renoise_is_reproducible(self) -> None:
        kwargs = dict(
            measurement=np.array([0.5, -0.25]),
            forward=lambda image: image,
            adjoint=lambda projection: projection,
            denoise=lambda state, sigma: state / (1.0 + sigma),
            sigma_levels=np.array([2.0, 0.5, 0.0]),
            gain=1.0,
            gamma_bar=0.2,
            cap=3.0,
            cg_iterations=2,
            chain_count=2,
            transition="renoise",
            seed=123,
        )
        first = reconstruct(**kwargs)
        second = reconstruct(**kwargs)
        np.testing.assert_array_equal(first.chains, second.chains)

    def test_deterministic_ode_transition_is_supported(self) -> None:
        result = reconstruct(
            np.array([1.0]),
            lambda image: image,
            lambda projection: projection,
            lambda state, sigma: np.zeros_like(state),
            np.array([1.0, 0.25, 0.0]),
            gain=1.0,
            gamma_bar=0.1,
            cap=2.0,
            transition="ode",
            seed=1,
        )
        self.assertEqual(result.reconstruction.shape, (1,))
        self.assertTrue(np.isfinite(result.reconstruction).all())


if __name__ == "__main__":
    unittest.main()
