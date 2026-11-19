# Simulation contract

This document describes the `xray-sim` acquisition path and its
`static_kinematic_gaussian_v1` crystallographic profile. It defines the
implemented model, its numerical estimates and its limits. The historical
`xrd-generate` renderer has separate relative-intensity conventions.

## Supported model

The crystallographic backend accepts exactly two distinct crystalline phases.
Each phase has an explicit cell, occupied atomic basis, coherent-domain size
and seeded orientation-distribution recipe. Neutral atomic form factors are
available for Cu, Ti, Fe and Nb; other elements are rejected. Nominal built-in
cells are `cu_fcc`, `ti_hcp`, `fe_bcc` and `nb_bcc`. They are example inputs,
not certified standards.

Material state is a bounded, piecewise-linear binary-mixture field along the
scan coordinate, extruded through the declared slab. A representative
orientation population supplies one static detector response per phase. Those
responses are mixed through footprint and exposure-time integration.

The population is not a set of spatially positioned grains. The diffraction
backend does not use lazy grain identities or compute beam/grain intersections.
It reuses the same representative orientation population throughout each phase.
Overlapping exposures share the deterministic phase field and texture, but no
overlap-induced grain covariance, changing grain population or parallax is
predicted. Conditional Poisson draws are independent for distinct readout
identities given their means.

The reference profile `synthetic_reference_fixture_v1` instead maps material
state weights to supplied or synthetic detector rates. It is useful for testing
acquisition and output handling; its channels do not implement physical XRD or
fluorescence transport.

## Units and coordinate conventions

| Quantity | Convention |
|---|---|
| Cell lengths, wavelength | Angstrom |
| Cell angles and texture spread parameters | Degrees |
| Reciprocal vectors | Cycles per angstrom, without a `2*pi` multiplier |
| Coherent-domain length | Nanometres |
| Sample positions, beam width, detector pitch/distance | Millimetres |
| Exposure time and velocity | Seconds and millimetres per second |
| Incident flux density | Photons per second per square micrometre |
| Representative illuminated volume | Cubic micrometres |
| Phase density | Grams per cubic centimetre |
| Native detector rates / integrated means | Events per second / expected events |
| Quaternions | Unit, scalar-first `wxyz`, active rotations |

The incident beam is along laboratory `+x`; the planar detector is `x = D`.
Detector columns increase along `y` and rows along `z`. The beam center is
specified in native `(row, column)` pixel-center coordinates. For outgoing unit
direction `n` and wavelength `lambda`:

```text
k_in = (1/lambda, 0, 0)
G(ray) = (n - (1, 0, 0))/lambda
dOmega = D * dA / r^3
```

Geometry is static at a common scattering origin. Sample translation changes
the mixture encountered by the beam, without relocating that origin.

Cell vectors are columns of the direct-cell matrix `A`; reciprocal vectors
are `A^(-T) hkl`. The full metric is retained for nonorthogonal cells.
Fractional atomic coordinates are in `[0,1)`, and colocated occupancies must not
sum above one. Structure factors sum the declared basis, including systematic
extinctions. Enumeration includes both Friedel signs; reflection multiplicity
is supplied by the enumerated vectors.

## Intensity and material amount

The isotropic coherent-domain approximation is a normalized
three-dimensional reciprocal-space Gaussian:

```text
K_sigma(delta_G) = (2*pi*sigma^2)^(-3/2)
                   * exp(-|delta_G|^2/(2*sigma^2))
integral K_sigma(delta_G) d^3G = 1
sigma = shape_factor / (10 * coherent_domain_nm * sqrt(8*ln(2)))
```

The reciprocal FWHM is the shape factor divided by the domain length in
angstroms. This approximates a finite-domain shape transform; it does not
resolve grain shape, strain or coherent interference between domains.

For a reflection `h`, representative phase volume `V`, cell volume `v_cell`
and structure factor `F_h` in electron units:

```text
d(sigma_scatter)/dOmega = r_e^2 * V/v_cell^2 * |F_h|^2
                         * K_sigma(G(ray)-G_h) * P(n)
pixel_rate = incident_flux_density
             * integral_pixel d(sigma_scatter)/dOmega dOmega
```

The code converts micrometre-based flux and volume to angstrom units before
applying this law. `K_sigma` has units of angstrom cubed, making the differential
cross-section an area. The pinned classical electron radius is
`2.8179403262e-5` angstrom. Orientation volume weights sum to one and partition
the same supplied volume. Supported polarization is unpolarized,
laboratory-`y` or laboratory-`z` linear polarization, or no correction.

`illuminated_volume_um3` supplies the amount normalization for a fully occupied
representative volume. It is not computed from slab dimensions, beam width or
grain intersections. The caller must choose a volume consistent with the
intended experiment. Partial sample occupancy scales the phase signal through
footprint integration. A wider beam at fixed supplied volume changes spatial
averaging, rather than automatically increasing the material amount.

Material mass fractions `w_i` are converted to volume fractions using the
basis/cell-derived phase densities:

```text
phi_i = (w_i/rho_i) / sum_j(w_j/rho_j)
local_rate = occupied * sum_i(phi_i * equal_volume_phase_rate_i)
             + detector_background_rate
```

Declared material composition and density must match the atomic basis and cell;
inconsistent definitions are rejected. Equal mass fractions generally do not
give equal volume fractions. The detector background persists when the beam
misses the sample and is scaled by exposure duration and shutter transmission.

There is no per-image normalization, empirical size multiplier, separate
excitation weight or additional Lorentz factor. Flux, supplied illuminated
volume and duration scale the phase counts linearly. For a narrow isolated
accessible reflection, its Haar orientation average integrated over outgoing
directions approaches `lambda^2/(2*|G_h|)`, independent of domain size at fixed
material amount. A single orientation at exact Bragg, a finite orientation
population or a cropped detector can change its collected intensity with domain
size. The powder result does not justify normalizing each rendered image.

## Texture

The supported orientation families are `random`, `fiber`, `partial_fiber`
and `component`, with an optional isotropic fraction. Seeded populations are
persistent across exposures, and every reflection of an orientation is rotated
together. Increasing the population changes the discrete approximation to the
declared orientation distribution; its convergence must be assessed separately.

`crystal_direction` and `sample_direction` are Cartesian directions in their
respective frames. They are not Miller-index triples for a general lattice.
Convert lattice directions through the direct-cell basis before supplying them,
particularly for nonorthogonal cells.

Crystal-to-sample orientations are followed by one active
`sample_to_lab_wxyz` rotation. Both phases must share this fixed sample pose.
Changing the pose rotates the same seeded specimen population instead of
redrawing it. Continuous sample rotation during an exposure is not supported.

## Pixel integration, clipping and accuracy

The detector integrates pixel areas with tensor-product Gauss-Legendre
quadrature. `pixel_subdivision` specifies nodes per pixel axis, with a smaller
`comparison_subdivision` for an independent-order difference estimate.
The backend supports at most 262144 native channels.

Support selection intersects reciprocal-space balls with the outgoing Ewald
sphere, then conservatively bounds the affected pixel areas. Peaks centered
outside the panel contribute when their tails reach the detector. Edge and
corner clipping removes the uncollected signal without redistributing it.
Reflection enumeration uses the detector pixel-area corners plus a reciprocal
support margin.

A preflight guard rejects unresolved peak widths. Both quadrature orders must
satisfy

```text
pixel_pitch_mm / (distance_mm * wavelength_A * nodes_per_axis) <= min(sigma)
```

This prevents silently accepting very narrow peaks missed by every node.
Passing it is not a convergence proof. The low-level projector's explicit
undersampling waiver is diagnostic only and is not enabled by the backend.

For the finite supplied peak list, an analytical complete-outgoing-sphere
integral bounds the signal outside the retained reciprocal support balls.
Positive strengths and polarization no greater than one make this also an upper
bound for the detector's omitted contribution. It does not bound tails of
reflections beyond the enumeration cutoff, orientation-population error or
physical-model uncertainty.

The phase-image estimate is `2*abs(fine - comparison)` plus the finite-list
support bound per pixel. Acquisition propagates it with adaptive spatial and
time integration estimates. A result is rejected if the combined estimate
exceeds `atol + rtol*abs(expected_mean)` or a work/depth/memory guard is exhausted.
These difference estimates are numerical diagnostics, not rigorous accuracy
bounds. Reflection cutoff, orientation population and model assumptions require
separate refinement or reference comparisons.

## Expectations, counts and saved truth

The API separates `expected(request)` from `observe(request)`. Clean
expectations contain floating-point expected native counts and numerical error
estimates. Observations contain nonnegative integer Poisson counts. The noise
identity does not change the material, orientation population or clean mean.

Expectation keys include material, crystal/atomic data, backend settings,
exposure parameters, source/runtime identity and requested accuracy. Caches
store clean means and their estimates, with checksummed disk records and bounded
steady-state storage. Atomic writes need temporary disk headroom; resource
settings are not whole-process memory or operating-system disk quotas.

The CLI uses `xray_generator.crystallographic_experiment/1` for crystallographic
configurations and `xray_generator.fixture_experiment/1` for reference-rate
configurations. Both write a shared `.runs/<id>.json` record containing normalized
configuration and runtime identity. Observation manifests use
`xray_generator.fixture_bundle/1` and reference a
`xray_generator.fixture_run/1` record; the profile identifies the scattering
model. These schema names are preserved for format compatibility.

`observable.npz` contains only `counts`. `private.npz` contains
`expected_mean` and `numerical_error`; the run record contains the compact
material and acquisition truth. Retain the run record with the observation
bundles, and keep private truth out of inference inputs when evaluating recovery.
`xray-sim inspect` verifies checksums and the referenced run record. Direct
low-level bundle writes without a run record are marked as lacking recoverable
training truth.

## Physical limits

The model has a dimensional absolute-intensity scale, without experimental
absolute calibration. Atomic form factors use a neutral IT92 subset on
`0 <= |G| < 4` inverse angstroms; see
[atomic-data provenance](atomic_data_provenance.md).

Absorption/self-absorption, physical fluorescence/XRF production and transport,
anomalous and thermal factors, diffuse scattering, multiple scattering,
dynamical diffraction, general strain and grain-shape transforms, source
bandwidth/divergence, detector efficiency/PSF, dead time, pileup, dynamic rotation
and material evolution are not implemented. The numerical checks do not certify
these omitted effects or agreement with measured XRD.
