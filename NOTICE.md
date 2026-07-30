# Notice

Copyright © 2026 the authors of *Acquisition-Adaptive Data Consistency for
Zero-Shot Diffusion CT Reconstruction*. The original software in this
repository is released under the MIT License.

This repository is a pre-publication research-code package. It contains an
original, framework-independent implementation of the acquisition-conditioned
data-consistency law, proximal conjugate-gradient primitives, reliability-mask
construction, and release validation tools.

The repository does **not** redistribute:

- the DM4CT source code or implementations of its comparison methods;
- the AAPM Low Dose CT Grand Challenge data;
- the LoDoInd data;
- the measured synchrotron Rocks data;
- pretrained diffusion checkpoints;
- reconstructed test images or internal experiment caches.

Those resources remain governed by their respective authors, repositories,
licenses, terms of use, and data-access conditions. DM4CT is treated as an
external research dependency because its public repository did not state a
redistribution license when this package was prepared.

The generic numerical routines in this repository were written independently
for this release. They are not copied from the external DM4CT, tomosipo,
ts_algorithms, DiffStateGrad, or baseline-sampler implementations.
