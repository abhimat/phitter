# Phitter
Python package to generate and fit binary models to observables. Uses [PHOEBE](http://www.phoebe-project.org) for generating observables (light curves, RVs) and [SPISEA](https://spisea.readthedocs.io/en/latest/) for generating stellar properties (e.g., masses, radii, bandpass luminosities). Provides classes for fitting observations using an MCMC sampler (e.g. [emcee](https://emcee.readthedocs.io/en/stable/index.html)).

Observables that can be modeled and fit to observations include:

* Photometry (default set: $K'$- and $H$-bands with the Keck II NIRC2 imager).
* Radial velocities of both primary and secondary star.



Near-term to do features:

* Allow flexible combinations of photometric filters for fitting observables.

