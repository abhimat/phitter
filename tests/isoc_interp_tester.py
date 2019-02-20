#!/usr/bin/env python

# Isochrone interpolation tester
# ---
# Abhimat Gautam

from phoebe_phitter import isoc_interp

import numpy as np


isochrone = isoc_interp.isochrone_mist(age=3.9e6, ext=2.63, dist=7.971e3, phase='MS')

interp_params_full, interp_params_lc = isochrone.rad_interp(5.)

print(interp_params_full)
print(interp_params_lc)

interp_params_full, interp_params_lc = isochrone.rad_interp(10.)

print(interp_params_full)
print(interp_params_lc)


interp_params_full, interp_params_lc = isochrone.mass_init_interp(5.)

print(interp_params_full)
print(interp_params_lc)

interp_params_full, interp_params_lc = isochrone.mass_init_interp(10.)

print(interp_params_full)
print(interp_params_lc)