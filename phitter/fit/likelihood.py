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
    Instantiate this object with an observations object to indicate the observed
    observables.
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
        
    
    def evaluate(self, model_observation):
        log_likelihood = -0.5 * np.sum(
            ((self.observations.obs - model_observation.obs) / \
             self.observations.obs_uncs)**2.
        )
        
        if np.isnan(log_likelihood):
            return -1e300
        else:
            return log_likelihood

class log_likelihood_chisq_weighted_obs_type(log_likelihood_chisq):
    """
    log_likelihood_chisq_weighted_obs_type is an object to obtain chi squared
    log likelihood, while weighing different observation types equally. This
    weighting scheme may be useful for cases where number of a given observation
    type is much larger than that of another.
    
    The output log likelihood is calculated in the following way:
    log_like_total = (log_like_t1)/(n_t1) + (log_like_t2)/(n_t2) + ...
    (see e.g., Lam+ 2022)
    
    Instantiate this object with an observations object to indicate the observed
    observables.
    """
    
    inputs = (
        'model_observation',
    )
    
    outputs = (
        'log_likelihood',
    )
    
    def evaluate(self, model_observation):
        log_likelihood = 0
        
        if self.observations.num_obs_phot > 0:
            log_likelihood += (-0.5 * np.sum(
                ((self.observations.obs_phot - model_observation.obs_phot) / \
                 self.observations.obs_uncs_phot)**2.
            )) / self.observations.num_obs_phot
        
        if self.observations.num_obs_rv > 0:
            log_likelihood += (-0.5 * np.sum(
                ((self.observations.obs_rv - model_observation.obs_rv) / \
                 self.observations.obs_uncs_rv)**2.
            )) / self.observations.num_obs_rv
        
        if np.isnan(log_likelihood):
            return -1e300
        else:
            return log_likelihood
    