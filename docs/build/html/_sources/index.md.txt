% phitter documentation master file.

# phitter

phitter is an open-source python package to simulate observables  from stellar binary systems and to fit them to observed data. Observables that can be calculated and fit with phitter include photometry (i.e., observed fluxes) and line-of-sight radial velocities (RVs).

Modeling of binary systems and calculation of observables is primarily handled with [PHOEBE](http://www.phoebe-project.org). When computing flux from model binaries, synthetic photometry for stars is derived for a wide range of telescope and passbands using [SPISEA](https://spisea.readthedocs.io/en/latest/). Parameters for the binary system's stellar components can be derived via interpolation of model stellar tracks (phitter currently implements [MIST](http://waps.cfa.harvard.edu/MIST/)). Otherwise, arbitrary stellar parameters for one or both stars can also be specified.

Fitting of observables to binary models is conducted with the use of MCMC sampling code. We provide support for sampling with nested sampling codes like [MultiNest](https://github.com/farhanferoz/MultiNest) (via [PyMultiNest](https://github.com/JohannesBuchner/PyMultiNest)), [UltraNest](https://johannesbuchner.github.io/UltraNest/), or [dynesty](https://dynesty.readthedocs.io/en/stable/index.html). Example scripts to demonstrate how to set up a fitter are provided.

\[Structure\]

\[How to set up a model binary\]

\[Including additional effects on photometric observables.\]

\[Including additional effects on RV observables.\]

\[How to derive own passbands and implement into PHOEBE\]

\[How to fit observables with nested sampling.\]

## Table of Contents

```{toctree}
:caption: 'Contents:'
:maxdepth: 2
```

# Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
