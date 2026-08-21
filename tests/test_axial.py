import unittest

import torch
from torch import nn

from robust_ct.axial import (
    EXPECTED_PARAMETER_COUNT,
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


def inputs(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(808)
    tweedie = torch.randn(batch, 3, 1, 12, 10, generator=generator)
    epsilon = torch.randn(batch, 3, 1, 12, 10, generator=generator)
    timestep = torch.tensor([25, 750], dtype=torch.long)[:batch]
    return tweedie, epsilon, timestep


class FixedResidual(nn.Module):
    def __init__(self, residual: torch.Tensor) -> None:
        super().__init__()
        self.residual = residual

    def forward(self, *_: torch.Tensor) -> torch.Tensor:
        return self.residual


class AxialRefinerTests(unittest.TestCase):
    def test_parameter_count_and_zero_initialized_output(self) -> None:
        torch.manual_seed(8)
        refiner = AxialCenterX0Refiner()
        self.assertEqual(parameter_count(refiner), EXPECTED_PARAMETER_COUNT)
        self.assertEqual(EXPECTED_PARAMETER_COUNT, 5_984)
        tweedie, epsilon, timestep = inputs()
        self.assertEqual(
            int(torch.count_nonzero(refiner(tweedie, epsilon, timestep))),
            0,
        )

    def test_neighbor_swap_and_replicated_center_identities(self) -> None:
        torch.manual_seed(9)
        refiner = AxialCenterX0Refiner()
        with torch.no_grad():
            refiner.zero_out.weight.normal_()
        tweedie, epsilon, timestep = inputs()
        swapped = torch.tensor([2, 1, 0])
        original = refiner(tweedie, epsilon, timestep)
        exchanged = refiner(tweedie[:, swapped], epsilon[:, swapped], timestep)
        self.assertTrue(torch.equal(original, exchanged))

        tweedie_center = tweedie[:, 1:2].expand(-1, 3, -1, -1, -1).clone()
        epsilon_center = epsilon[:, 1:2].expand(-1, 3, -1, -1, -1).clone()
        replicated = refiner(tweedie_center, epsilon_center, timestep)
        self.assertTrue(torch.equal(replicated, torch.zeros_like(replicated)))

    def test_raw_baseline_remains_unclipped(self) -> None:
        value = torch.tensor([[[[-2.0, -0.25, 0.5, 2.0]]]])
        raw = affine_normalize_x0(value, lower=-1.0, upper=1.0)
        clipped = normalize_x0(value, lower=-1.0, upper=1.0)
        self.assertTrue(torch.equal(raw, value))
        self.assertTrue(
            torch.equal(clipped, torch.tensor([[[[-1.0, -0.25, 0.5, 1.0]]]]))
        )

        tweedie, epsilon, timestep = inputs(batch=1)
        base = torch.full((1, 1, 12, 10), 1.7)
        residual = torch.full_like(base, -0.2)
        candidate = candidate_center_x0_normalized(
            FixedResidual(residual),
            base,
            tweedie,
            epsilon,
            timestep,
        )
        self.assertTrue(torch.equal(candidate, base + residual))

    def test_physical_residual_and_two_sigma_conversion(self) -> None:
        dtype = torch.float64
        center = torch.tensor([[[[0.8, -0.1]]]], dtype=dtype)
        epsilon = torch.tensor([[[[0.3, -0.2]]]], dtype=dtype)
        delta_normalized = torch.tensor([[[[0.4, -0.1]]]], dtype=dtype)
        delta_x0 = delta_x0_norm_to_physical(
            delta_normalized,
            lower=-1.0,
            upper=1.0,
        )
        sigma_ddpm = torch.tensor(0.2, dtype=dtype)
        sigma_schedule = torch.tensor(0.7, dtype=dtype)
        corrected = corrected_epsilon_from_delta_x0(
            epsilon,
            delta_x0,
            sigma_ddpm=sigma_ddpm,
        )
        anchor = refined_center_anchor(
            center,
            epsilon,
            delta_x0,
            sigma_schedule=sigma_schedule,
            sigma_ddpm=sigma_ddpm,
        )
        self.assertTrue(
            torch.allclose(anchor, center - sigma_schedule * corrected, atol=1e-14)
        )

        normalized = affine_normalize_x0(center, lower=-1.0, upper=1.0)
        recovered = normalized_x0_to_physical(
            normalized,
            lower=-1.0,
            upper=1.0,
        )
        self.assertTrue(torch.allclose(recovered, center))

    def test_bridge_uses_ddpm_sigma_not_karras_sigma(self) -> None:
        class QuarterResidual(nn.Module):
            def forward(self, feature, epsilon, timestep):
                del epsilon, timestep
                return torch.full_like(feature[:, 1], 0.25)

        sigmas = torch.linspace(0.1, 1.1, 1000)
        bridge = X0ToEpsilonResidualBridge(
            QuarterResidual(),
            sigmas,
            lower=-0.4,
            upper=1.4,
        )
        feature = torch.zeros(1, 3, 1, 4, 4)
        epsilon = torch.full_like(feature, 0.2)
        residual = bridge(feature, epsilon, torch.tensor([400]))
        expected_delta_x0 = 0.25 * 0.9
        expected = torch.full_like(residual, -expected_delta_x0 / sigmas[400])
        self.assertTrue(torch.allclose(residual, expected, atol=1e-7, rtol=0.0))


if __name__ == "__main__":
    unittest.main()
