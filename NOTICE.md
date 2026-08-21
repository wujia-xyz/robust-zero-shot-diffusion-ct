# Notice

Copyright © 2026 the authors of *Acquisition-Conditioned Axial Prior Refinement
for Zero-Shot Diffusion CT Reconstruction*. The original software in this
repository is released under the MIT License.

This repository is a pre-publication research-code package. It contains an
original implementation of the acquisition-conditioned controller, compact
axial x0 refiner, two-noise-scale bridge, three-state own-slice sampler,
proximal conjugate-gradient primitives, reliability-mask construction, and
release validation tools.

The repository does **not** redistribute:

- the DM4CT source code or implementations of its comparison methods;
- the AAPM Low Dose CT Grand Challenge data;
- the LoDoInd data;
- the measured synchrotron Rocks data;
- pretrained diffusion checkpoints;
- the trained axial-refiner checkpoint;
- reconstructed test images or internal experiment caches.

Those resources remain governed by their respective authors, repositories,
licenses, terms of use, and data-access conditions. DM4CT is treated as an
external research dependency because its public repository did not state a
redistribution license when this package was prepared.

The generic numerical routines in this repository were written independently
for this release. They are not copied from the external DM4CT, tomosipo,
ts_algorithms, DiffStateGrad, or baseline-sampler implementations.
