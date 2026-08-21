import unittest

import torch
from torch import nn

from robust_ct.axial import X0ToEpsilonResidualBridge
from robust_ct.triplet import (
    FixedObservationUpdate,
    SliceObservation,
    TripletSamplerConfig,
    apply_own_slice_dc,
    controller_weight,
    sample_center_only,
    sample_triplet,
    torch_proximal_cg,
    triplet_prior_step,
)


class ZeroRefiner(nn.Module):
    def forward(self, feature, epsilon, timestep):
        del epsilon, timestep
        return torch.zeros_like(feature[:, 1])


class CenterResidual(nn.Module):
    def forward(self, feature, epsilon, timestep):
        del epsilon, timestep
        neighbor_difference = feature[:, 0] - feature[:, 2]
        return 0.1 * neighbor_difference


def ddpm_schedules() -> tuple[torch.Tensor, torch.Tensor]:
    sigma = torch.linspace(0.01, 2.0, 1000)
    alpha = 1.0 / (1.0 + sigma.square())
    return alpha, sigma


def observations() -> tuple[SliceObservation, SliceObservation, SliceObservation]:
    return (
        SliceObservation(torch.full((1, 1, 2, 2), 1.0), "z-1"),
        SliceObservation(torch.full((1, 1, 2, 2), 2.0), "z"),
        SliceObservation(torch.full((1, 1, 2, 2), 3.0), "z+1"),
    )


class TripletTests(unittest.TestCase):
    def test_controller_formula(self) -> None:
        value = controller_weight(
            0.5,
            gamma_bar=0.1,
            g_cap=0.25,
            gain=20.0,
            denominator_epsilon=0.0,
        )
        self.assertEqual(value, 5.0)

    def test_prior_step_changes_center_anchor_only(self) -> None:
        alpha, sigma_ddpm = ddpm_schedules()
        states = torch.tensor(
            [[[[[0.5, 0.5], [0.5, 0.5]]],
              [[[0.2, 0.2], [0.2, 0.2]]],
              [[[-0.5, -0.5], [-0.5, -0.5]]]]]
        )
        bridge = X0ToEpsilonResidualBridge(
            CenterResidual(),
            sigma_ddpm,
            lower=-1.0,
            upper=1.0,
        ).eval().requires_grad_(False)
        step = triplet_prior_step(
            states,
            sigma_schedule=0.5,
            timestep=250,
            alpha_bar_t=alpha[250],
            base_forward=lambda value, timestep: torch.zeros_like(value),
            epsilon_bridge=bridge,
            lower=-1.0,
            upper=1.0,
            require_replicated_identity=False,
        )
        self.assertTrue(torch.equal(step.anchors[:, 0], states[:, 0]))
        self.assertTrue(torch.equal(step.anchors[:, 2], states[:, 2]))
        self.assertFalse(torch.equal(step.anchors[:, 1], states[:, 1]))

    def test_own_slice_dc_never_fuses_measurements(self) -> None:
        priors = torch.zeros(1, 3, 1, 2, 2)
        calls: list[float] = []

        def solver(prior, measurement, gamma, iterations):
            del gamma, iterations
            calls.append(float(measurement.mean()))
            return prior + measurement

        result = apply_own_slice_dc(
            priors,
            observations(),
            step=0,
            gamma=1.0,
            cg_iterations=6,
            observation_update=FixedObservationUpdate(),
            dc_solvers=(solver, solver, solver),
        )
        self.assertEqual(calls, [1.0, 2.0, 3.0])
        self.assertEqual(result.solver_call_count, 3)
        self.assertTrue(torch.equal(result.images[:, 1], torch.full((1, 1, 2, 2), 2.0)))

    def test_complete_triplet_trajectory_and_center_output(self) -> None:
        alpha, sigma_ddpm = ddpm_schedules()
        bridge = X0ToEpsilonResidualBridge(
            ZeroRefiner(),
            sigma_ddpm,
            lower=-1.0,
            upper=1.0,
        ).eval().requires_grad_(False)
        template = torch.zeros(1, 3, 1, 2, 2)
        config = TripletSamplerConfig(
            num_levels=2,
            cg_iterations=1,
            gamma_bar=0.1,
            g_cap=1.0,
            gain=1.0,
            lower=-1.0,
            upper=1.0,
        )

        def solver(prior, measurement, gamma, iterations):
            del prior, gamma, iterations
            return measurement.clone()

        result = sample_triplet(
            initial_template=template,
            observations=observations(),
            sigma_steps=torch.tensor([1.0, 0.25, 0.0]),
            alpha_cumprod=alpha,
            sigma_ddpm=sigma_ddpm,
            config=config,
            base_forward=lambda value, timestep: torch.zeros_like(value),
            epsilon_bridge=bridge,
            seeds=(100, 101, 102),
            dc_solvers=(solver, solver, solver),
        )
        self.assertEqual(result.dc_call_count, 6)
        self.assertEqual(result.noise_draw_count, 3)
        self.assertTrue(result.replicated_center_identity)
        self.assertTrue(
            torch.equal(result.center, torch.full((1, 1, 2, 2), 2.0))
        )

    def test_boundary_path_is_single_state_and_adapter_free(self) -> None:
        alpha, sigma_ddpm = ddpm_schedules()
        config = TripletSamplerConfig(
            num_levels=2,
            cg_iterations=1,
            gamma_bar=0.1,
            g_cap=1.0,
            gain=1.0,
            lower=-1.0,
            upper=1.0,
        )

        def solver(prior, measurement, gamma, iterations):
            del prior, gamma, iterations
            return measurement.clone()

        result = sample_center_only(
            initial_template=torch.zeros(1, 1, 2, 2),
            observation=SliceObservation(torch.full((1, 1, 2, 2), 4.0), "z"),
            sigma_steps=torch.tensor([1.0, 0.25, 0.0]),
            alpha_cumprod=alpha,
            sigma_ddpm=sigma_ddpm,
            config=config,
            base_forward=lambda value, timestep: torch.zeros_like(value),
            seed=99,
            dc_solver=solver,
        )
        self.assertEqual(result.dc_call_count, 2)
        self.assertEqual(result.noise_draw_count, 3)
        self.assertTrue(torch.equal(result.center, torch.full((1, 1, 2, 2), 4.0)))

    def test_torch_proximal_cg_identity_operator(self) -> None:
        prior = torch.zeros(1, 1, 2, 2)
        measurement = torch.full_like(prior, 2.0)
        solution = torch_proximal_cg(
            prior,
            measurement,
            gamma=1.0,
            iterations=2,
            forward=lambda value: value,
            adjoint=lambda value: value,
        )
        self.assertTrue(torch.allclose(solution, torch.ones_like(solution)))


if __name__ == "__main__":
    unittest.main()
