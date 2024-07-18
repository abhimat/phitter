# Priors for fitters in phitter
# ---
# Abhimat Gautam

import numpy as np
from scipy import stats

class uniform_prior(object):
    """
    Uniform distribution prior
    
    Parameters
    ----------
    bound_lo : float
        Lower bound on the distribution 
    bound_up : float
        Upper bound on the distribution
    """
    
    def __init__(self, bound_lo, bound_up):
        self.bound_lo = bound_lo
        self.bound_up = bound_up
        
        self.param_count = 1
        
        return
    
    def __call__(self, cube):
        return (cube * (self.bound_up - self.bound_lo)) + self.bound_lo
    
    def __repr__(self):
        return f'<uniform_prior: bound_lo {self.bound_lo} bound_up {self.bound_up}>'

class gaussian_prior(object):
    """
    Gaussian / normal distribution prior
    
    Parameters
    ----------
    mean : float
        Mean of the distribution
    sigma : float
        Sigma of the distribution
    """
    
    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma
        
        self.param_count = 1
        
        return
    
    def __call__(self, cube):
        return stats.norm.ppf(
            cube,
            loc=self.mean, scale=self.sigma,
        )
    
    def __repr__(self):
        return f'<gaussian_prior: mean {self.mean} sigma {self.sigma}>'

class const_prior(object):
    """
    Constant value prior
    
    Parameters
    ----------
    value : float
        Constant value to return
    """
    
    def __init__(self, value):
        self.value = value
        
        self.param_count = 1
        
        return
    
    def __call__(self, cube):
        return self.value
    
    def __repr__(self):
        return f'<const_prior: value {self.value}>'

class prior_collection(object):
    """
    Collection of prior objects. Transformation from unit cube to parameter
    space takes place with the prior_transform() function. Contains separate
    prior transform functions for use with different sampling software.
    
    Parameters
    ----------
    priors_list : list[prior]
        List of priors that consitute the full set of parameters being modeled.
    """
    
    def __init__(self, priors_list):
        self.priors_list = priors_list
        
        # Calculate number of parameters from the priors
        self.num_params = 0
        
        for prior in self.priors_list:
            self.num_params += prior.param_count
        
        return
    
    def prior_transform_multinest(self, cube, ndim, nparam):
        """
        Prior transform function for use with PyMultiNest
        """
        for i in range(nparam):
            cube[i] = self.priors_list[i](cube[i])
        
        # TODO: Add support for dependent priors
    
    def prior_transform_ultranest(self, cube):
        """
        Prior transform function for use with Ultranest
        """
        
        params = cube.copy()
        
        for i in range(self.num_params):
            params[i] = self.priors_list[i](cube[i])
        
        return params
    
    def prior_transform_dynesty(self, u):
        """
        Prior transform function for use with Dynesty
        """
        
        params = np.array(u)
        
        for i in range(self.num_params):
            params[i] = self.priors_list[i](u[i])
        
        return params
    