#!/usr/bin/env python

# Light curve generation testing
# ---
# Abhimat Gautam

from phoebe_phitter import lc_calc

from phoebe import u
from phoebe import c as const

import numpy as np

# # Test stellar model parameters (detached)
plot_name = 'det_det'

star1_mass = 38.2 * u.solMass
star1_rad = 45.0 * u.solRad
star1_teff = 28260. * u.K
star1_mag_Kp = 10.17
star1_mag_H = 12.30

star1_params = (star1_mass, star1_rad, star1_teff, star1_mag_Kp, star1_mag_H)

star2_mass = 38.2 * u.solMass
star2_rad = 45.0 * u.solRad
star2_teff = 28260. * u.K
star2_mag_Kp = 10.17
star2_mag_H = 12.30

star2_params = (star2_mass, star2_rad, star2_teff, star2_mag_Kp, star2_mag_H)

# # Test stellar model parameters (semidetached)
# plot_name = 'sd_det'
#
# star1_mass = 38.2 * u.solMass
# star1_rad = 50.0 * u.solRad
# star1_teff = 28260. * u.K
# star1_mag_Kp = 10.17
# star1_mag_H = 12.30
#
# star1_params = (star1_mass, star1_rad, star1_teff, star1_mag_Kp, star1_mag_H)
#
# star2_mass = 38.2 * u.solMass
# star2_rad = 45.0 * u.solRad
# star2_teff = 28260. * u.K
# star2_mag_Kp = 10.17
# star2_mag_H = 12.30
#
# star2_params = (star2_mass, star2_rad, star2_teff, star2_mag_Kp, star2_mag_H)

# # Test stellar model parameters (contact)
# plot_name = 'of_det'
#
# star1_mass = 38.2 * u.solMass
# star1_rad = 55.0 * u.solRad
# star1_teff = 28260. * u.K
# star1_mag_Kp = 10.17
# star1_mag_H = 12.30
#
# star1_params = (star1_mass, star1_rad, star1_teff, star1_mag_Kp, star1_mag_H)
#
# star2_mass = 38.2 * u.solMass
# star2_rad = 45.0 * u.solRad
# star2_teff = 28260. * u.K
# star2_mag_Kp = 10.17
# star2_mag_H = 12.30
#
# star2_params = (star2_mass, star2_rad, star2_teff, star2_mag_Kp, star2_mag_H)

# Test stellar model parameters (contact)
plot_name = 'det_of'

star2_mass = 38.2 * u.solMass
star2_rad = 55.0 * u.solRad
star2_teff = 28260. * u.K
star2_mag_Kp = 10.17
star2_mag_H = 12.30

star2_params = (star1_mass, star1_rad, star1_teff, star1_mag_Kp, star1_mag_H)

star1_mass = 38.2 * u.solMass
star1_rad = 45.0 * u.solRad
star1_teff = 28260. * u.K
star1_mag_Kp = 10.17
star1_mag_H = 12.30

star1_params = (star2_mass, star2_rad, star2_teff, star2_mag_Kp, star2_mag_H)

# # Test stellar model parameters (both components semidetached)
# plot_name = 'sd_sd'
#
# star1_mass = 38.2 * u.solMass
# star1_rad = 50.0 * u.solRad
# star1_teff = 28260. * u.K
# star1_mag_Kp = 10.17
# star1_mag_H = 12.30
#
# star1_params = (star1_mass, star1_rad, star1_teff, star1_mag_Kp, star1_mag_H)
#
# star2_mass = 38.2 * u.solMass
# star2_rad = 50.0 * u.solRad
# star2_teff = 28260. * u.K
# star2_mag_Kp = 10.17
# star2_mag_H = 12.30
#
# star2_params = (star2_mass, star2_rad, star2_teff, star2_mag_Kp, star2_mag_H)

# Binary system parameters
binary_period = 20.0 * u.d
binary_ecc = 0.0
binary_inc = 90.0 * u.deg
binary_t0 = 0.0

binary_params = (binary_period, binary_ecc, binary_inc, binary_t0)


# Observation times
observation_times = (np.array([0.0, 5.0, 10.0, 15.0]), np.array([0.0, 5.0, 10.0, 15.0]))



print(lc_calc.binary_star_lc(star1_params, star2_params, binary_params, observation_times, use_blackbody_atm=True, make_mesh_plots=True, plot_name=plot_name, print_diagnostics=True))
