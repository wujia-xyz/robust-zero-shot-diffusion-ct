# Acquisition-Conditioned Axial Prior Refinement for Zero-Shot Diffusion CT Reconstruction

Official code repository for:

> **Acquisition-Conditioned Axial Prior Refinement for Zero-Shot Diffusion CT
> Reconstruction**
>
> Jia Wu, Xiaoming Jiang, Hongying Meng, Yamei Luo, and Zhangyong Li

The method combines an acquisition-conditioned data-consistency controller
with a compact axial refiner. It uses a frozen two-dimensional diffusion prior,
requires no paired projection/target training data, and reconstructs every
interior location through three physical states with three separate
measurements. The axial refiner changes only the center prior anchor, and only
the center reconstruction is retained.

## Method

For an interior center location (z), the reverse state is
(mathcal X_t=\{x_{z-1,t},x_{z,t},x_{z+1,t}\}). The frozen 2-D prior predicts
(epsilon_{	heta,j,t}) for each (j\in\{z-1,z,z+1\}). Two clean-image
coordinates are then formed:

\[
\widehat x^{2\mathrm D}_{0,j,t}
=x_{j,t}-\sigma_{\mathrm{DDPM},t}\epsilon_{\theta,j,t},
\qquad
D^{2\mathrm D}_{j,t}
=x_{j,t}-s_t\epsilon_{\theta,j,t}.
\]

The first is the Tweedie estimate used by the axial refiner; the second is the
anchor used by the continuous Karras sampler. The 5,984-parameter refiner
receives the normalized Tweedie estimates, noise-prediction features, and DDPM
timestep from all three states. It predicts a center-only normalized-x0
residual. After conversion to physical image units, the center anchor becomes

\[
\widetilde D_{z,t}
=D^{2\mathrm D}_{z,t}
+\frac{s_t}{\sigma_{\mathrm{DDPM},t}}\Delta x_{0,t},
\]

while the neighboring anchors remain unchanged.

The acquisition-conditioned controller supplies one coefficient shared by the
three own-slice systems:

\[
g_{\mathrm{cap}}
=\frac{c}{\mathrm{GAIN}}\left(1+\frac{\sigma_n^2}{\sigma_{n,0}^2}\right),
\qquad
\gamma_t
=\mathrm{GAIN}\min\left\{
\frac{\bar\gamma}{s_t^2+10^{-8}},g_{\mathrm{cap}}
\right\}.
\]

Each physical state is updated with its own measurement, operator, and
reliability matrix:

\[
x^{\mathrm{prox}}_{j,t}
=\arg\min_x\frac{1}{2}
\lVert W_j^{1/2}(A_jx-\widetilde y_{j,t})\rVert_2^2
+\frac{\gamma_t}{2}\lVert x-D_{j,t}\rVert_2^2.
\]

Independent noise is added to form the next three reverse states. At volume
boundaries, the implementation uses a true single-state path and bypasses the
axial refiner.

## Repository contents

```text
src/robust_ct/axial.py       axial x0 refiner and two-sigma bridge
src/robust_ct/triplet.py     triplet prior, own-slice DC, and boundary sampler
src/robust_ct/law.py         acquisition-conditioned controller
src/robust_ct/proximal.py    NumPy proximal conjugate-gradient solver
src/robust_ct/reliability.py detector-column reliability mask
src/robust_ct/schedule.py    Karras and controller schedules
configs/current_method.json  current architecture and operating points
results_summary/             complete-volume descriptive statistics
scripts/                     result export and public-release validation
tests/                       numerical and method-contract tests
```

The CT interfaces are callback based. The same method core can therefore be
connected to ASTRA, torch-radon, tomosipo, or another matched projection
backend without embedding third-party sampler code in this repository.

## Installation

The controller and NumPy numerical core require Python 3.10 or later:

```bash
git clone https://github.com/wujia-xyz/robust-zero-shot-diffusion-ct.git
cd robust-zero-shot-diffusion-ct
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Install the complete axial/triplet implementation with PyTorch support:

```bash
python -m pip install -e ".[test,axial]"
python -m pytest
python scripts/validate_release.py
```

The supplied `environment.yml` provides an equivalent Conda environment.

## Core API

The acquisition-conditioned coefficient can be evaluated independently of the
CT backend:

```python
from robust_ct import compute_normalized_cap
from robust_ct.triplet import controller_weight

g_cap = compute_normalized_cap(
    noise_variance=sigma_n2,
    prior_error=sigma_n0,
    gain=gain,
    calibration_factor=c,
)
gamma_t = controller_weight(
    sigma_schedule=sigma_t,
    gamma_bar=gamma_bar,
    g_cap=g_cap,
    gain=gain,
)
```

The axial module exposes the exact released architecture and the bridge between
the DDPM and Karras noise coordinates:

```python
from robust_ct.axial import AxialCenterX0Refiner, X0ToEpsilonResidualBridge

refiner = AxialCenterX0Refiner()
refiner.load_state_dict(refiner_state)
refiner.eval().requires_grad_(False)

bridge = X0ToEpsilonResidualBridge(
    refiner,
    sigma_ddpm,
    lower=image_lower,
    upper=image_upper,
).eval().requires_grad_(False)
```

`sample_triplet` accepts this bridge, the frozen 2-D prior callback, three own
measurements, and three single-slice DC callbacks. `sample_center_only`
implements the endpoint path without constructing artificial neighbors.
`BoundedRankOneObservationUpdate` implements the measured-data detector-profile
update used for Rocks.

The current operating points, architecture hashes, 100-level sampler settings,
and six-CG update are recorded in
[`configs/current_method.json`](configs/current_method.json).

## Data and pretrained priors

The experiments use the data definitions and clean-image priors from DM4CT.
Obtain each resource from its official host and follow its terms of use.

| Domain | Data | Pixel-space diffusion prior |
|---|---|---|
| AAPM | [Low Dose CT Grand Challenge](https://www.aapm.org/grandchallenge/lowdosect/) | [lodochallenge_pixel_diffusion](https://huggingface.co/jiayangshi/lodochallenge_pixel_diffusion) |
| LoDoInd | [LoDoInd](https://zenodo.org/records/10391412) | [lodoind_pixel_diffusion](https://huggingface.co/jiayangshi/lodoind_pixel_diffusion) |
| Rocks | [Synchrotron Rocks](https://zenodo.org/records/15420527) | [synchrotron_pixel_diffusion](https://huggingface.co/jiayangshi/synchrotron_pixel_diffusion) |

The comparison protocol and external reconstruction methods are available from
[DM4CT](https://github.com/DM4CT/DM4CT). This repository contains the original
method implementation but does not redistribute controlled data, third-party
prior weights, the trained axial-refiner checkpoint, or external baseline
implementations. `configs/paths.example.toml` shows the expected local path
layout, and `configs/current_method.json` records the refiner checkpoint hashes.

## Complete-volume results

The following values are the current one-trajectory results reported in the
paper. They are unweighted means over every image in each evaluation volume.

| Domain | Configuration | Count | PSNR (dB) | SSIM |
|---|---:|---:|---:|---:|
| AAPM | i | 526 | 31.74 | 0.87 |
| AAPM | ii | 526 | 27.74 | 0.78 |
| AAPM | iii | 526 | 29.69 | 0.82 |
| AAPM | iv | 526 | 29.42 | 0.82 |
| AAPM | v | 526 | 31.06 | 0.86 |
| LoDoInd | i | 500 | 23.33 | 0.66 |
| LoDoInd | ii | 500 | 19.98 | 0.55 |
| LoDoInd | iii | 500 | 23.44 | 0.68 |
| LoDoInd | iv | 500 | 22.15 | 0.63 |
| LoDoInd | v | 500 | 21.48 | 0.61 |
| Rocks | 200 views | 660 | 33.00 | 0.61 |
| Rocks | 100 views | 660 | 32.37 | 0.57 |
| Rocks | 60 views | 660 | 31.93 | 0.55 |

Full-precision means, sample standard deviations, minima, and maxima are stored
in [`results_summary/current_method_summary.json`](results_summary/current_method_summary.json).
The file contains 7,110 reconstructions across the 13 acquisition settings.

## Citation

The manuscript is under review. Until final bibliographic information is
available, cite the entry in [`CITATION.cff`](CITATION.cff).

## License

Original software in this repository is released under the
[MIT License](LICENSE). Data, pretrained models, and external software remain
subject to their respective terms. See [`NOTICE.md`](NOTICE.md).
