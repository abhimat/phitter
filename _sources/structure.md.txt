% Structure of Phitter

# Structure of Phitter

To get started with using Phitter, it helps to understand the structure of the package. It’s built around 3 main tasks needed to be carried out when simulating a binary light curve or fitting a light curve to data, represented in the package layout:

1. {py:obj}`phitter.params`: This portion of the package consists of setting up the parameters for a binary system. It consists of stellar parameters for each component star in the binary (e.g., mass or radius), along with the parameters specifying the structure of the binary system (e.g., orbital period, inclination of orbit). Importantly, the stellar parameters can be set up arbitrarily, or instead interpolated from a stellar isochrone.
2. {py:obj}`phitter.calc`: The calculation of observables (photometric fluxes or radial velocities) is carried out in this portion of the package. This stage also includes functions to allow *bulk* shifts to the flux or radial velocities, such as reddening from extinction or adding in a center of mass radial velocity.
3. {py:obj}`phitter.fit`: Fitting modeled observables to observed values is handled by this part of the package. This portion of the package allows interfacing with sampling codes like emcee or UltraNest. It includes functions to calculate likelihoods and provides convenience functions for transformations of priors.

In addition to these broad areas of Phitter, the package includes a {py:obj}`phitter.observables` object that facilitates storing observations and associated times, types, and passbands. The package also includes {py:obj}`phitter.filters` which represent observation passbands.