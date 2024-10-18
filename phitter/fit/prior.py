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

class multivariate_gaussian_prior(object):
    """
    Multivariate Gaussian / normal distribution prior
    
    Parameters
    ----------
    means : np.array(dtype=float)
        Means of the distribution for each parameter
    sigmas : np.array(dtype=float)
        Sigmas of the distribution for each parameter
    covar : np.array(dtype=float)
        Covariance matrix between the quantities, of shape
        [parameter x parameter]. Assume this has been calculated for the
        quantities after normalization. i.e.: calculated for each quantity
        after: (quant - mean(quant))/sigma(quant)
    """
    
    def __init__(self, means, sigmas, covar):
        self.means = means
        self.sigmas = sigmas
        self.covar = covar
        
        num_params = len(means)
        
        self.param_count = num_params
        
        # Perform necessary matrix calculations in order to calculate
        # rotation matrix,
        # following method described by Johannes Buchner at
        # https://johannesbuchner.github.io/UltraNest/priors.html
        
        a = np.linalg.inv(covar)
        l, v = np.linalg.eigh(a)
        rotation_matrix = np.dot(v, np.diag(1. / np.sqrt(l)))
        
        self.a_mat = a
        self.l_mat = l
        self.v_mat = v
        self.rotation_matrix = rotation_matrix
        
        return
    
    def __call__(self, cube):
        independent_gaussian = stats.norm.ppf(cube)
        
        # Use rotation matrix to transform multidimensional gaussian,
        # following method described by Johannes Buchner at
        # https://johannesbuchner.github.io/UltraNest/priors.html
        
        return self.means + self.sigmas*np.einsum(
            'ij,kj->ki',
            self.rotation_matrix,
            independent_gaussian,
        )
    
    def __repr__(self):
        return f'<multivariate_gaussian_prior:\nmeans {self.means} sigmas {self.sigmas} covar {self.covar} >'

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
        
        cur_param_index = 0
        
        for prior_index in range(self.num_params):
            # Determine number of parameters current prior object is handling
            prior_num_params = self.priors_list[prior_index].num_params
            
            cube[cur_param_index:cur_param_index + prior_num_params] =\
                self.priors_list[prior_index](
                    cube[cur_param_index:cur_param_index + prior_num_params],
                )
            
            cur_param_index += prior_num_params
    
    def prior_transform_ultranest(self, cube):
        """
        Prior transform function for use with Ultranest
        """
        
        params = cube.copy()
        
        cur_param_index = 0
        
        for prior_index in range(self.num_params):
            # Determine number of parameters current prior object is handling
            prior_num_params = self.priors_list[prior_index].num_params
            
            params[cur_param_index:cur_param_index + prior_num_params] =\
                self.priors_list[prior_index](
                    cube[cur_param_index:cur_param_index + prior_num_params],
                )
            
            cur_param_index += prior_num_params
        
        return params
    
    def prior_transform_dynesty(self, u):
        """
        Prior transform function for use with Dynesty
        """
        
        params = np.array(u)
        
        cur_param_index = 0
        
        for prior_index in range(self.num_params):
            # Determine number of parameters current prior object is handling
            prior_num_params = self.priors_list[prior_index].num_params
            
            params[cur_param_index:cur_param_index + prior_num_params] =\
                self.priors_list[prior_index](
                    u[cur_param_index:cur_param_index + prior_num_params],
                )
            
            cur_param_index += prior_num_params
        
        return params
    