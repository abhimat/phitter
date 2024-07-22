% Phitter documentation master file.

# phitter

[Phitter](https://github.com/abhimat/phitter) is an open-source python package to simulate observables  from stellar binary systems and to fit them to observed data. Observables that can be calculated and fit with phitter include photometry (i.e., observed fluxes) and line-of-sight radial velocities (RVs).

Modeling of binary systems and calculation of observables is primarily handled with [PHOEBE](http://www.phoebe-project.org). When computing flux from model binaries, synthetic photometry for stars is derived for a wide range of telescope and passbands using [SPISEA](https://spisea.readthedocs.io/en/latest/). Parameters for the binary system's stellar components can be derived via interpolation of model stellar tracks (Phitter currently implements [MIST](http://waps.cfa.harvard.edu/MIST/)). Otherwise, arbitrary stellar parameters for one or both stars can also be specified.

Fitting of observables to binary models is conducted with the use of MCMC sampling code. We provide support for sampling with nested sampling codes like [UltraNest](https://johannesbuchner.github.io/UltraNest/), [MultiNest](https://github.com/farhanferoz/MultiNest) (via [PyMultiNest](https://github.com/JohannesBuchner/PyMultiNest)), or [dynesty](https://dynesty.readthedocs.io/en/stable/index.html). Phitter can also be used with non-nested sampling MCMC codes like [emcee](https://emcee.readthedocs.io/en/stable/). Examples to demonstrate how to set up a fitter are provided for UltraNest, dynesty, and emcee.

<!-- ## Getting started

\[Installation\]

[Structure of Phitter](structure)

### Tutorials


```{automodapi} phitter.fit
``` -->
   
## Documentation Contents
```{toctree}
:maxdepth: 2

structure
tutorials
```

## Indices and tables
- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
