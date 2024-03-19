# Functions to perform adjustments to radial velocities
# ---
# Abhimat Gautam

import numpy as np
from phoebe import u

def apply_com_velocity(bin_observables, com_velocity):
    """
    Modeled observables are calculated with a center of mass velocity of 0 km/s.
    This function allows applying a constant center of mass velocity to RVs.
    
    Parameters
    ----------
    bin_observables : observables
        observables object containing RVs to which to apply center of mass
        velocity.
    com_velocity : Astropy Quantity, velocity unit
        Binary system's center of mass velocity.
    
    Returns
    -------
    observables
        observables object, where RVs have center of mass velocity added.
    """
    # Return if there are no RV observations
    if bin_observables.num_obs_rv == 0:
        return bin_observables
    
    bin_observables.obs[bin_observables.obs_rv_filter] += \
        com_velocity.to(u.km/u.s).value
    
    return bin_observables
