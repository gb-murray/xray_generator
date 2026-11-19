# XRD Generator

Generate two-dimensional X-ray diffraction data from explicit crystal structures
and texture distributions. The `xray_generator` package combines static
crystallographic phase responses with compact material recipes, moving-beam
exposures, cached expectations, and native Poisson counts.

The supported crystallographic profile is `static_kinematic_gaussian_v1`. It
uses normalized reciprocal-space Gaussian peaks, atomic structure factors, and
pixel-area detector integration. It currently represents a two-phase,
homogenized material field with a fixed orientation population per phase.
Absolute intensity has a dimensional model scale but has not been experimentally
calibrated. Physical XRF simulation is not implemented.

## Install

Use Python 3.11 or 3.12. From a checkout, in your Python environment:

```bash
python -m pip install -e .
```

Installation provides the `xray-sim` and `xrd-generate` commands.

## Interactive walkthrough

The [end-to-end notebook](examples/demo.ipynb) walks through
material definition, crystallographic texture, moving exposures, cached means,
Poisson observations, count-conserving polar views, and recoverable data exports.
It also includes small reference-rate and lazy-geometry examples.

```bash
python -m pip install jupyterlab ipykernel
python -m jupyter lab examples/end_to_end_simulation.ipynb
```

Choose the Python environment containing the installed package and run all cells.
The default uses a 512 by 512 detector; generated outputs stay under the ignored
`datasets/notebook_demo/` directory. The notebook source is saved without outputs.

## Run a simulation

The textured example defines nominal FCC Cu and HCP Ti cells, persistent
orientation populations, a 512 by 512 detector, and moving exposures across a
material interface.

```bash
xray-sim validate configs/v1/crystallographic_texture_demo.json
xray-sim expected configs/v1/crystallographic_texture_demo.json --output datasets/texture_demo --cache-dir datasets/texture_cache
xray-sim observe configs/v1/crystallographic_texture_demo.json --output datasets/texture_demo --cache-dir datasets/texture_cache
```

`validate` checks configuration consistency and reports capabilities without
rendering a detector. `expected` computes clean expected counts and numerical
error estimates. `observe` draws native Poisson counts using the configured
readout seeds. Unsupported inputs, unresolved peak widths, and exhausted
accuracy or work budgets cause the request to fail.

For a fast acquisition and file-format example, use the synthetic reference
configuration:

```bash
xray-sim observe configs/v1/reference_scan_fixture.json --output datasets/reference_demo --cache-dir datasets/reference_cache
```

Reference rates exercise acquisition, caching and readout; they do not predict
physical XRD or XRF intensities.

## Output and reproducibility

An `expected` run writes one `.npz` file per expectation with `total_mean` and
`numerical_error`. An `observe` run writes one checksummed directory per
exposure/readout identity:

| File | Contents |
|---|---|
| `observable.npz` | Nonnegative integer detector `counts`, suitable as simulated observations. |
| `private.npz` | Clean `expected_mean` and `numerical_error`; keep these separate from model inputs when evaluating inference. |
| `manifest.json` | Profile, identities, checksums, calibration flag and a reference to the run record. |
| `.runs/<id>.json` in the output root | Shared normalized material, acquisition, scattering and readout configuration, runtime identity and expectation IDs. |

`xray-sim inspect <bundle-directory>` verifies an observation bundle and its
referenced run record. The versioned wire schemas retain the names
`xray_generator.fixture_bundle/1` and `xray_generator.fixture_run/1`; the
`profile` field distinguishes crystallographic data from synthetic fixtures.
Keep the output root's `.runs` directory with its bundles to retain reconstructable
simulation truth.

Identical requests reuse clean expectations. Changing a readout replicate gives
new noise without redrawing the material or texture. Changes to physical
settings, source/runtime identity or requested accuracy invalidate incompatible
cache entries. Completed observation bundles are verified on reuse. Counts and
clean arrays are never normalized per image.

See the [simulation contract](docs/simulation.md) for units, material mixing,
texture conventions, numerical estimates and physical limits, and
[atomic-data provenance](docs/atomic_data_provenance.md) for the bundled tables.

## Compatibility

The historical `src.*` API and `xrd-generate` command retain their existing
relative-intensity behavior and configuration format:

```bash
xrd-generate configs/fcc_texture_grain_example.json
```

See the [legacy configuration guide](docs/configuration.md). Its relative
intensities and broadening options are distinct from the `xray-sim`
crystallographic profile.

## Development

Install the test dependencies and run the suite:

```bash
python -m pip install -e ".[test]"
python -m pytest
```

For independent crystallographic comparisons, install the optional pinned Gemmi
reference with `python -m pip install -e ".[validation]"`. These checks are
skipped without Gemmi; it is not a simulation runtime dependency.

## License

The simulator is distributed under [Apache-2.0](LICENSE). The selected Gemmi
atomic-data subset retains its separate MPL-2.0 license and attribution, as
described in [atomic-data provenance](docs/atomic_data_provenance.md).
