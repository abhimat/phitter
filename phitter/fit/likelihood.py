# Likelihoods for fitters in phitter
# ---
# Abhimat Gautam

import phoebe
from phoebe import u
from phoebe import c as const
import numpy as np
from astropy.modeling import Model
from phitter import observations

class log_likelihood_chisq(Model):
    """
    log_likelihood_chisq is an object to obtain chi squared log likelihood.
    Instantiate this object with an observations object
    """
    
    inputs = (
        'model_obs_time',
        'model_obs',
        'model_filt',
        'model_obs_type',
    )
    
    outputs = (
        'log_likelihood',
    )
    
    def __init__(self, observations, *args, **kwargs):
        super(log_likelihood_chisq, self).__init__(*args, **kwargs)
        
        # observations is going to be an observations object