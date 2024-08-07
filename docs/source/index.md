% Phitter documentation master file.

# phitter

[Phitter](https://github.com/abhimat/phitter) is an open-source python package to simulate observables  from stellar binary systems and to fit them to observed data. Observables that can be calculated and fit with phitter include photometry (i.e., observed fluxes) and line-of-sight radial velocities (RVs).

Modeling of binary systems and calculation of observables is primarily handled with [PHOEBE](http://www.phoebe-project.org). When computing flux from model binaries, synthetic photometry for stars is derived for a wide range of telescope and passbands using [SPISEA](https://spisea.readthedocs.io/en/latest/). Parameters for the binary system's stellar components can be derived via interpolation of model stellar tracks (Phitter currently implements [MIST](http://waps.cfa.harvard.edu/MIST/)). Otherwise, arbitrary stellar parameters for one or both stars can also be specified.

Fitting of observables to binary models is conducted with the use of MCMC sampling code. We provide support for sampling with nested sampling codes like [UltraNest](https://johannesbuchner.github.io/UltraNest/) (tutorial [here](notebooks/fit_with_ultranest)), [MultiNest](https://github.com/farhanferoz/MultiNest) (via [PyMultiNest](https://github.com/JohannesBuchner/PyMultiNest)), or [dynesty](https://dynesty.readthedocs.io/en/stable/index.html). Phitter can also be used with non-nested sampling MCMC codes like [emcee](https://emcee.readthedocs.io/en/stable/).

## Citing Phitter

[![DOI](https://zenodo.org/badge/170761219.svg)](https://zenodo.org/doi/10.5281/zenodo.8370775)

If you use Phitter in published research, please cite the [software repository via zenodo](https://zenodo.org/doi/10.5281/zenodo.8370775). Please also cite the paper [Gautam et al. (2024)](https://ui.adsabs.harvard.edu/abs/2024ApJ...964..164G/abstract) where many of Phitter's functionalities are described.

Phitter is built atop the functionality offered by [PHOEBE 2](https://ui.adsabs.harvard.edu/abs/2016ApJS..227...29P/abstract) and [SPISEA](https://ui.adsabs.harvard.edu/abs/2020AJ....160..143H/abstract) which should also be cited in research using Phitter.

## Documentation Contents
```{toctree}
:maxdepth: 1

features
installation
structure
tutorials
apidocs/index
```

## Indices and search
- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
