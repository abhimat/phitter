# phoebe_phitter
Python package to generate and fit binary models to observables. Uses [PHOEBE](http://www.phoebe-project.org) for generating observables (light curves, RVs) and [SPISEA](https://spisea.readthedocs.io/en/latest/) for generating stellar properties (e.g., masses, radii, bandpass luminosities). Provides classes for fitting observations using an MCMC sampler (e.g. [emcee](https://emcee.readthedocs.io/en/stable/index.html)).

Currently set up for fitting observables at near-infrared K'- and H-bands (at the Keck II NIRC2 imager).