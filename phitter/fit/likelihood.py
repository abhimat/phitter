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
        'model_observation',
    )
    
    outputs = (
        'log_likelihood',
    )
    
    def __init__(self, observations, *args, **kwargs):
        super(log_likelihood_chisq, self).__init__(*args, **kwargs)
        
        # Save observations to the object
        self.observations = observations
        
        if self.observations.obs_uncs == None:
           self.observations.obs_uncs = np.ones_like(self.observations.obs)
        
    
    def evaluate(model_observation):
        log_likelihood = -0.5 * np.sum(
            ((self.observations.obs - model_observation.obs) / \
             self.observations.obs_uncs)**2.
        )
        
        if np.isnan(log_likelihood):
            return -1e300
        else:
            return log_likelihood