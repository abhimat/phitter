# Likelihoods for fitters in phitter
# ---
# Abhimat Gautam

import numpy as np

class log_likelihood_chisq(object):
    """
    log_likelihood_chisq is an object to obtain chi squared log likelihood.
    Instantiate this object with an observables object to indicate the observed
    observables.
    
    Parameters
    ----------
    observations : observables
        observables object, specified with the observations to be fit.
    """
    
    def __init__(self, observations, *args, **kwargs):
        super(log_likelihood_chisq, self).__init__(*args, **kwargs)
        
        # Save observations to the object
        self.observations = observations
        
        if self.observations.obs_uncs is None:
           self.observations.obs_uncs = np.ones_like(self.observations.obs)
        
    
    def evaluate(self, model_observables):
        log_likelihood = -0.5 * np.sum(
            ((self.observations.obs - model_observables.obs) / \
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
    
    Instantiate this object with an observables object to indicate the observed
    observables.
    
    Parameters
    ----------
    observations : observables
        observables object, specified with the observations to be fit.
    """
    
    def evaluate(self, model_observables):
        log_likelihood = 0
        
        if self.observations.num_obs_phot > 0:
            log_likelihood += (-0.5 * np.sum(
                ((self.observations.obs_phot - model_observables.obs_phot) / \
                 self.observations.obs_uncs_phot)**2.
            )) / self.observations.num_obs_phot
        
        if self.observations.num_obs_rv > 0:
            log_likelihood += (-0.5 * np.sum(
                ((self.observations.obs_rv - model_observables.obs_rv) / \
                 self.observations.obs_uncs_rv)**2.
            )) / self.observations.num_obs_rv
        
        if np.isnan(log_likelihood):
            return -1e300
        else:
            return log_likelihood

class log_likelihood_chisq_weighted_filts_and_obs_type(log_likelihood_chisq):
    """
    log_likelihood_chisq_weighted_filts_and_obs_type is an object to obtain chi
    squared log likelihood, while weighing different observation types, and
    filters within each observation type equally.
    This weighting scheme may be useful for cases where number of a given
    observation type or filt is much larger than that of another.
    
    The output log likelihood is calculated in the following way:
    log_like_total = (log_like_t1)/(n_t1) + (log_like_t2)/(n_t2) + ...
    (see e.g., Lam+ 2022)
    
    Instantiate this object with an observables object to indicate the observed
    observables.
    
    Parameters
    ----------
    observations : observables
        observables object, specified with the observations to be fit.
    """
    
    def evaluate(self, model_observables):
        log_likelihood = 0
        
        if self.observations.num_obs_phot > 0:
            # Add likelihood for each filt in photometry
            for cur_filt in self.observations.unique_filts_phot:
                # Get search filter for the current photometry filt
                cur_filt_search_filter = self.observations.phot_filt_filters[cur_filt]
                
                # Cut out relevant observations from observations and model
                cur_filt_obs = self.observations.obs[cur_filt_search_filter]
                cur_filt_obs_uncs = self.observations.obs_uncs[cur_filt_search_filter]
                cur_filt_model_obs = model_observables.obs[cur_filt_search_filter]
                
                cur_filt_num_obs = len(cur_filt_obs)
                
                if cur_filt_num_obs > 0:
                    # Compute likelihood and add to total
                    log_likelihood += (-0.5 * np.sum(
                        ((cur_filt_obs - cur_filt_model_obs) / \
                         cur_filt_obs_uncs)**2.
                    )) / cur_filt_num_obs
                else:
                    continue
        
        if self.observations.num_obs_rv > 0:
            # Add likelihood for each filt in RV
            for cur_filt in self.observations.unique_filts_rv:
                # Get search filter for the current RV filt
                cur_filt_search_filter = self.observations.rv_filt_filters[cur_filt]
                
                # Cut out relevant observations from observations and model
                cur_filt_obs = self.observations.obs[cur_filt_search_filter]
                cur_filt_obs_uncs = self.observations.obs_uncs[cur_filt_search_filter]
                cur_filt_model_obs = model_observables.obs[cur_filt_search_filter]
                
                cur_filt_num_obs = len(cur_filt_obs)
                
                if cur_filt_num_obs > 0:
                    # Compute likelihood and add to total
                    log_likelihood += (-0.5 * np.sum(
                        ((cur_filt_obs - cur_filt_model_obs) / \
                         cur_filt_obs_uncs)**2.
                    )) / cur_filt_num_obs
                else:
                    continue
        
        if np.isnan(log_likelihood):
            return -1e300
        else:
            return log_likelihood
            