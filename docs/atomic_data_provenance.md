# Atomic scattering data provenance

The crystallographic profile evaluates the neutral-atom IT92
four-Gaussian-plus-constant approximation:

```text
f0(s) = sum_i a_i * exp(-b_i*s^2) + c
s = |G|/2
```

`G` is in cycles per angstrom. The declared domain is
`0 <= s < 2` inverse angstroms, equivalently `0 <= |G| < 4`.
No ionic correction, anomalous dispersion, Debye-Waller factor or absorption
correction is supplied implicitly.

## Coefficient source and license

The Cu, Ti, Fe and Nb numerical rows are copied without renormalization from
the versioned [Gemmi v0.7.5 IT92 table](https://raw.githubusercontent.com/project-gemmi/gemmi/v0.7.5/include/gemmi/it92.hpp).
The complete upstream source SHA-256 is
`3e113254f8f43276afb202b36894034120b15d22c756def707f86c541cbe6d48`.
The source identifies the International Tables for Crystallography Volume C
1992-or-later parametrization and equivalent CCP4/cctbx tables.

The bundled [subset JSON](../src/xray_generator/physics/data/it92_subset.json)
records the source, convention and exact coefficients. Its byte hash is included
in each phase fingerprint and consequently the backend dependency key.
Coefficients are used directly; no rescaling to force `f0(0) = Z` is applied.

Upstream copyright is 2019 Global Phasing Ltd. The selected data file is
distributed under MPL-2.0; the complete upstream license is retained in
[GEMMI_LICENSE.txt](../src/xray_generator/physics/data/GEMMI_LICENSE.txt).
This attribution applies to that data subset, not a relicensing of the
independently implemented simulator. Gemmi itself is not required at runtime;
it is an optional independent test reference.

## Atomic masses and nominal cells

Standard atomic weights are the nominal NIST values Cu 63.546, Ti 47.867,
Fe 55.845 and Nb 92.90637. See the NIST atomic-weight pages for
[Cu](https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=Cu),
[Ti](https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=Ti),
[Fe](https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=Fe), and
[Nb](https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=Nb).

The mass conversion pins the CODATA 2018 atomic mass constant
`1.66053906660e-24 g`. Density is derived from occupied unit-cell mass divided
by the declared cell volume. The supplied demonstration cells are nominal
inputs, not certified lattice-parameter or density standards. See the
[simulation contract](simulation.md) for amount and intensity conventions.
