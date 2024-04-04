# Legacy functions for compatibility

import phoebe
from phoebe import u
from phoebe import c as const
import numpy as np
from phitter import filters

# Filters for default filter list
kp_filt = filters.nirc2_kp_filt()
h_filt = filters.nirc2_h_filt()

def phased_obs(observation_times, binary_period, t0,
               filts_list=[kp_filt, h_filt]):
    """Phase observation times to a given binary period and t0
    """
    num_filts = len(filts_list)
    
    # Read in observation times, and separate photometry from RVs
    filt_MJDs = observation_times[:num_filts]
    rv_MJDs = observation_times[num_filts]
    
    out_phased_obs = ()
    # Phase photometric observation times
    for filt_index in range(num_filts):
        # Phase the current filter's MJDs
        cur_filt_MJDs = filt_MJDs[filt_index]
        cur_filt_phased_days = (((cur_filt_MJDs - t0) %
                                 binary_period.to(u.d).value) /
                                binary_period.to(u.d).value)
        
        # Compute phase sorted inds
        cur_filt_phases_sorted_inds = np.argsort(cur_filt_phased_days)
        
        # Compute model times sorted to phase sorted inds
        cur_filt_model_times = cur_filt_phased_days *\
                               binary_period.to(u.d).value
        cur_filt_model_times = cur_filt_model_times[cur_filt_phases_sorted_inds]
        
        # Append calculated values to output tuple
        out_phased_obs = out_phased_obs + \
                         ((cur_filt_phased_days, cur_filt_phases_sorted_inds,
                           cur_filt_model_times), )
    
    # Phase RV observation times
    rv_phased_days = (((rv_MJDs - t0) % binary_period.to(u.d).value) /
                      binary_period.to(u.d).value)
    
    rv_phases_sorted_inds = np.argsort(rv_phased_days)
    
    rv_model_times = (rv_phased_days) * binary_period.to(u.d).value
    rv_model_times = rv_model_times[rv_phases_sorted_inds]
    
    # Append RV values to output tuple
    out_phased_obs = out_phased_obs + \
                     ((rv_phased_days, rv_phases_sorted_inds,
                       rv_model_times), )
    
    return out_phased_obs
