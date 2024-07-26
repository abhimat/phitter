% Phitter Tutorials

# Tutorials

Some basic tutorials to understand how to use Phitter's main features.

## [Simulate a model binary](notebooks/simulate_model_binary)

This tutorial walks through the basics of setting up a simulation of a model binary system, with models of observable fluxes and RVs.

## [Simulate a model binary using stellar parameters from an isochrone](notebooks/simulate_model_binary_wMIST_star_params)

This tutorial introduces how to use Phitter to specify stellar parameters for binary components from MIST isochrones. Constraining stellar parameters to those derived from isochrones can be useful if the age of a binary or the host star population is well constrained.

## [Including additional effects on photometric fluxes](notebooks/add_phot_effects)

By default, Phitter's model binaries are located at a distance of 10 parsecs and simulated fluxes have no extinction applied. In order to match observables to data, additional photometric effects can be included. Examples include reddening from extinction, distance modulus, or flux zero-point uncertainties.

This tutorial walks through how to apply a distance modulus and reddening from an extinction law in order to obtain simulated photometry with 

## [Including additional effects on radial velocities](notebooks/add_rv_effects)

Similar to photometric data, additional effects can be included on radial velocities (RVs) simulated from Phitter. Currently this only includes a constant offset for the system's center of mass.

## Advanced: how to incorporate new photometric passbands into Phitter

***tutorial under construction*** 

## [Fitting binary model to observed data using nested sampling via Ultranest](notebooks/fit_with_ultranest)

This tutorial walks through the process of fitting a binary star model to observation data using nested sampling with [UltraNest](https://johannesbuchner.github.io/UltraNest/).

***in progress***

## [Fitting binary model to observed data using emcee](notebooks/fit_with_emcee)

This tutorial walks through the process of fitting a binary star model to observation data using MCMC sampling with [emcee](https://emcee.readthedocs.io).

```{toctree}
:caption: 'Contents:'
:maxdepth: 1

notebooks/simulate_model_binary
notebooks/simulate_model_binary_wMIST_star_params
notebooks/add_phot_effects
notebooks/add_rv_effects
notebooks/create_mock_data
notebooks/fit_with_ultranest
notebooks/fit_with_emcee
```