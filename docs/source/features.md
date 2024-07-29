% Features of Phitter

# Features

* Quick, easy setup of model binary systems ([tutorial](notebooks/simulate_model_binary))
* Easily integrate stellar parameters from model isochrones ([tutorial](notebooks/simulate_model_binary_wMIST_star_params))
* Easily fit stellar models to observations of binary systems using different samplers (tutorial with [UltraNest](notebooks/fit_with_ultranest) and [emcee](notebooks/fit_with_emcee))

## Why use Phitter with PHOEBE, rather than just PHOEBE alone?

PHOEBE is a powerful and complex package to simulate binary systems. However, the complexity also makes it challenging to get started with or use in certain contexts. Phitter offers advantages in a few notable areas:

* **Easy setup of binary systems with wide variety of observables.** Set up of binary systems in Phitter is [simple and straightforward](notebooks/simulate_model_binary). Adding in different sets of observables is also easy to do in Phitter, with easy addition of model fluxes and RVs in multiple passbands.
* **Easy to generate synthetic photometry for model systems.** Phitter's model photometry is built on top of [SPISEA](https://spisea.readthedocs.io/en/latest/), which offers the ability to calculate synthetic stellar photometry with a wide variety of photometric passbands and model atmospheres. Furthermore, complex extinction laws can be integrated into SPISEA, and subsequently used to derive reference fluxes for model stars in Phitter's model binary systems.
* **Easily change between detached, semi-detached, and contact binary systems.** Changing between detached, semi-detached, and contact binary systems requires separate setups in PHOEBE. Phitter is able to automatically switch between these binary system setups as necessary when generating model observables and performing fits.
* **Easy integration with sampling codes for fitting observations.** Phitter allows easy interfacing with sampling codes to fit models to observations, including [with emcee](notebooks/fit_with_emcee) or [with nested samplers like UltraNest](notebooks/fit_with_ultranest).


While Phitter works well with a wide variety of binary systems, including eccentric systems or binary systems including a compact object, more complex systems may require switching to using PHOEBE directly. Some examples of systems or physical phenomena that are not yet supported in Phitter include triple systems, stellar spots, or precessing binary orbits. If you have a suggestion of setup that may work well in Phitter, [make a suggestion as an issue on Github](https://github.com/abhimat/phitter/issues)!
