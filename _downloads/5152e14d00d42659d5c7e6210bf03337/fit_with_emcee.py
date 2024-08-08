#!/usr/bin/env python

import os
# Turn off parallelisation in phoebe
# Needs to be done *before* phoebe is imported
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PHOEBE_ENABLE_MPI"] = "FALSE"

from phitter import observables, filters
from phitter.params import star_params, binary_params, isoc_interp_params
from phitter.calc import model_obs_calc, phot_adj_calc, rv_adj_calc
from phitter.fit import likelihood, prior

import numpy as np

from phoebe import u
from phoebe import c as const
import matplotlib as mpl
import matplotlib.pyplot as plt

import pickle
from astropy.table import Table

import emcee
from multiprocessing import Pool

# Set up filters
filter_153m = filters.hst_f153m_filt()
filter_127m = filters.hst_f127m_filt()

# Read observations data
with open('./mock_obs_table.pkl', 'rb') as in_file:
    obs_table = pickle.load(in_file)

# Set up observables objects
# Model observables object, which only contains times and types of observations
model_observables = observables.observables(
    obs_times=obs_table['obs_times'].data,
    obs_filts=obs_table['obs_filts'].data, obs_types=obs_table['obs_types'].data,
)

# An observables object for the observations, used when computing likelihoods
observations = observables.observables(
    obs_times=obs_table['obs_times'].data, obs=obs_table['obs'].data, obs_uncs=obs_table['obs_uncs'].data,
    obs_filts=obs_table['obs_filts'].data, obs_types=obs_table['obs_types'].data,
)

# Make stellar parameters object
isoc_stellar_params_obj = isoc_interp_params.isoc_mist_stellar_params(
    age=8e9,
    met=0.0,
    use_atm_func='merged',
    phase='RGB',
    ext_Ks=2.2,
    dist=8e3*u.pc,
    filts_list=[filter_153m, filter_127m],
    ext_law='NL18',
)

# Make binary params object
bin_params = binary_params.binary_params()

# Set up a binary model object
binary_model_obj = model_obs_calc.binary_star_model_obs(
    model_observables,
    use_blackbody_atm=False,
    print_diagnostics=False,
)

# Set up likelihood object for fitting parameters
log_like_obj = likelihood.log_likelihood_chisq(
    observations
)

def emcee_log_like(model_params):
    (
        star1_radius,
        star2_radius,
        bin_period,
        bin_inc,
        bin_t0,
        bin_rv_com,
        ext_153m,
        ext_alpha,
    ) = model_params
    
    # Obtain stellar params by interpolating along the isochrone
    star1_params = isoc_stellar_params_obj.interp_star_params_rad(
        star1_radius,
    )
    star2_params = isoc_stellar_params_obj.interp_star_params_rad(
        star2_radius,
    )
    
    # Set binary params
    bin_params.period = bin_period * u.d
    bin_params.inc = bin_inc * u.deg
    bin_params.t0 = bin_t0
    
    # Run binary model
    modeled_observables = binary_model_obj.compute_obs(
        star1_params, star2_params, bin_params,
        num_triangles=300,
    )
    
    # Check for situation where binary model fails
    # (i.e., unphysical conditions not able to be modeled)
    if np.isnan(modeled_observables.obs_times[0]):
        return -np.inf
    
    # Apply distance modulus
    # (We're assuming we know the distance, but this can be a fit parameter as well)
    modeled_observables = phot_adj_calc.apply_distance_modulus(
        modeled_observables,
        7.971e3*u.pc,
    )
    
    # Apply extinction
    modeled_observables = phot_adj_calc.apply_extinction(
        modeled_observables,
        2.2, filter_153m,
        ext_153m,
        isoc_red_law='NL18',
        ext_alpha=ext_alpha,
    )
    
    # Add RV center of mass velocity
    modeled_observables = rv_adj_calc.apply_com_velocity(
        modeled_observables,
        bin_rv_com * u.km / u.s,
    )
    
    # Compute and return log likelihood
    log_like = log_like_obj.evaluate(modeled_observables)
    
    return log_like

def emcee_log_prior(model_params):
    (
        star1_radius,
        star2_radius,
        bin_period,
        bin_inc,
        bin_t0,
        bin_rv_com,
        ext_153m,
        ext_alpha,
    ) = model_params
    
    s1_checks = 10.0 < star1_radius < 25.0
    
    s2_checks = 8.0 < star2_radius < 15.0
    
    bin_param_checks = 22.0 < bin_period < 28.0 and\
        40.0 < bin_inc < 140.0 and\
        53_795.0 < bin_t0 < 53_805.0 and\
        100.0 < bin_rv_com < 200.0
    
    ext_checks = 4.0 < ext_153m < 6.0 and\
        2.17 < ext_alpha < 2.29
    
    if s1_checks and s2_checks and bin_param_checks and ext_checks:
        log_prior = 0.0
        
        return log_prior
    else:
        return -np.inf

def emcee_log_prob(model_params):
    lp = emcee_log_prior(model_params)
    
    if not np.isfinite(lp):
        return -np.inf
    
    return lp + emcee_log_like(model_params)

if __name__ == '__main__':
    test_params = (
        18., 10.,
        23., 85.,
        53_801.,
        140.,
        4.4, 2.22,
    )
    
    log_like_test = emcee_log_prob(test_params)
    print(f'log prob from test parameters = {log_like_test:.3f}')
    
    # Initialize walker positions
    n_params = len(test_params)
    n_walkers = n_params * 10

    p0 = [(
        14 + 1.e0 * np.random.randn(),
        11 + 1.e0 * np.random.randn(),
        24 + 1.e0 * np.random.randn(),
        90. + ((10 + 5e0*np.random.randn()) * np.sign(np.random.rand() - 0.5)),
        53_801 + 1.e0 * np.random.randn(),
        140. + 5.e0 * np.random.randn(),
        4.4 + 1.e-1 * np.random.randn(),
        2.22 + 1.e-2 * np.random.randn(),
    ) for i in range(n_walkers)]
    
    # Set up backend to save run
    filename = 'emcee_chains.h5'
    backend = emcee.backends.HDFBackend(filename)
    
    # Set up sampler
    num_cores = 9
    mp_pool = Pool(num_cores)
    sampler = emcee.EnsembleSampler(
        n_walkers, n_params, emcee_log_prob,
        backend=backend, 
        pool=mp_pool,
    )
    
    # Determine number of steps completed so far
    n_steps=1000
    
    n_steps_completed = backend.iteration
    n_steps_remaining = n_steps - n_steps_completed
    
    if n_steps_completed == 0:
        sampler.run_mcmc(p0, n_steps_remaining, progress=True)
    elif n_steps_remaining > 0:
        sampler.run_mcmc(None, n_steps_remaining, progress=True)
    
    # Close multiprocessing pool
    mp_pool.close()
    