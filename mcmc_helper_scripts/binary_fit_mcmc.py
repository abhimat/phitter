#!/usr/bin/env python

# Binary light curve fitter
# Using PopStar with MIST tracks and PHOEBE binary models
# ---
# Abhimat Gautam

# Parellization optimization for emcee
import os
os.environ["OMP_NUM_THREADS"] = "1"

# Imports
import numpy as np
from scipy.optimize import leastsq

from phoebe_phitter import mcmc_fit

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.ticker import MultipleLocator

from multiprocessing import Pool

import cPickle as pickle
import sys

# Print out working directory
print(os.getcwd())

trial_num = 0
trial_num_walkers = 16
trial_burn_len = 0
trial_samp_len = 10

# trial_num = 1
# trial_num_walkers = 500
# trial_burn_len = 0
# trial_samp_len = 2000
# 
# trial_num = 2
# trial_num_walkers = 200
# trial_burn_len = 0
# trial_samp_len = 5000
#
# trial_num = 3
# trial_num_walkers = 500
# trial_burn_len = 0
# trial_samp_len = 5000

early_iters_cutoff = 200
early_iters_num_triangles = 200
final_iters_num_triangles = 500


# Isochrone parameters
isoc_age = 10e9
isoc_ext = 2.63
isoc_dist = 7.971e3
isoc_phase = 'RGB'
isoc_met = 0.0

# Read in observation data
target_star = 'S2-36'
with open('./lc_data.pkl', 'rb') as input_pickle:
    target_binary_period = pickle.load(input_pickle)
    phase_shift_fit = pickle.load(input_pickle)
    kp_target_mags = pickle.load(input_pickle)
    kp_target_mag_errors = pickle.load(input_pickle)
    kp_target_MJDs = pickle.load(input_pickle)
    kp_phased_days = pickle.load(input_pickle)
    h_target_mags = pickle.load(input_pickle)
    h_target_mag_errors = pickle.load(input_pickle)
    h_target_MJDs = pickle.load(input_pickle)
    h_phased_days = pickle.load(input_pickle)


# Initial binary parameters

init_Kp_ext_t = 2.85
init_H_ext_mod_t = -0.3
init_star1_rad_t = 37.0
init_star2_rad_t = 14.40
init_binary_inc_t = 90.
init_binary_period_t = 78.82818
init_binary_ecc_t = 0.0
init_binary_dist_t = 7971
init_t0_t = 53779.1987

# Set up MCMC fitting object
mcmc_fit_obj = mcmc_fit.mcmc_fitter_rad_interp()

## Make isochrone that will be interpolated
mcmc_fit_obj.make_isochrone(isoc_age, isoc_ext, isoc_dist, isoc_phase,
                            isoc_met, use_atm_func='phoenix')

## Set observation times used during fitting (in MJDs)
mcmc_fit_obj.set_observation_times(kp_target_MJDs, h_target_MJDs)

## Set observation mags, to compare with model mags
mcmc_fit_obj.set_observation_mags(
    kp_target_mags, kp_target_mag_errors,
    h_target_mags, h_target_mag_errors)

## Set number of triangles to use in model mesh
mcmc_fit_obj.set_model_numTriangles(early_iters_num_triangles)

## Set to use blackbody atmosphere
mcmc_fit_obj.set_model_use_blackbody_atm(True)

## Set to model H extinction modifier
mcmc_fit_obj.set_model_H_ext_mod(True)

## Set to not model eccentricity
mcmc_fit_obj.set_model_eccentricity(False)

## Set to model distance
mcmc_fit_obj.set_model_distance(False)
mcmc_fit_obj.default_dist = 7.971e3

# Set prior bounds
mcmc_fit_obj.set_Kp_ext_prior_bounds(1.0, 4.0)
# mcmc_fit_obj.set_H_ext_mod_prior_bounds(-2.0, 2.0)
mcmc_fit_obj.set_H_ext_mod_extLaw_sig_prior_bounds(5.0)

mcmc_fit_obj.set_period_prior_bounds(73.0, 85.0)
mcmc_fit_obj.set_dist_prior_bounds(4000., 12000.)
mcmc_fit_obj.set_t0_prior_bounds(init_t0_t - (init_binary_period_t * 0.5),
                                 init_t0_t + (init_binary_period_t * 0.5))

# Set up Emcee
def emcee_lnprob(theta):
    return mcmc_fit_obj.lnprob(theta)

ndim, nwalkers = 7, trial_num_walkers

p0 = [(init_Kp_ext_t + (2.e-1)*(np.random.rand() - 0.5),
       init_H_ext_mod_t + (2.e-1)*(np.random.rand() - 0.5),
       init_star1_rad_t + (5.e-1)*(np.random.rand() - 0.5),
       init_star2_rad_t + (7.e-1)*(np.random.rand() - 0.5),
       90. + ((90. - init_binary_inc_t + (5.e1)*(np.random.rand() - 0.5)) * np.sign(np.random.rand() - 0.5)),
       init_binary_period_t + (1.e-2)*(np.random.rand() - 0.5),
       init_t0_t + (4.e-1)*(np.random.rand() - 0.5)) for i in range(nwalkers)]

test_theta = (init_Kp_ext_t, init_H_ext_mod_t,
              init_star1_rad_t, init_star2_rad_t,
              init_binary_inc_t, init_binary_period_t,
              init_t0_t)

## Example call
import time

print('---')
print("Timing example binary light curve iteration")

start_time = time.time()

print("lnlike: {0:.3f}".format(mcmc_fit_obj.lnprob(test_theta)))

end_time = time.time()

print("Run time: {0:.3f} s".format(end_time - start_time))
print('---')


import emcee

### Make chains directory if it doesn't exist yet
if not os.path.exists('chains'):
    os.makedirs('chains')

filename = './chains/chains_try{0}.h5'.format(trial_num)
backend = emcee.backends.HDFBackend(filename)

mp_pool = Pool()
sampler = emcee.EnsembleSampler(nwalkers, ndim, emcee_lnprob, backend=backend, pool=mp_pool)

# If no iterations run yet, run on existing number of triangles with p0
nsteps_completed = backend.iteration
if nsteps_completed == 0:
    nsteps_torun = np.min([early_iters_cutoff - nsteps_completed, trial_samp_len - nsteps_completed])
    
    print('Running {0} iterations with {1} triangles'.format(
        nsteps_torun, early_iters_num_triangles))
    sampler.run_mcmc(p0, nsteps_torun, progress=True)

# If early iteration, run on existing number of triangles
nsteps_completed = backend.iteration
if nsteps_completed < early_iters_cutoff:
    nsteps_torun = np.min([early_iters_cutoff - nsteps_completed, trial_samp_len - nsteps_completed])
    
    print('Running {0} iterations with {1} triangles'.format(
        nsteps_torun, early_iters_num_triangles))
    sampler.run_mcmc(None, nsteps_torun, progress=True)

# If early iterations completed, switch to higher number of triangles and run remaining trials    
nsteps_completed = backend.iteration
if nsteps_completed >= early_iters_cutoff:
    ## Switch to higher number of triangles
    mcmc_fit_obj.set_model_numTriangles(final_iters_num_triangles)
    
    print('Running {0} iterations with {1} triangles'.format(
        trial_samp_len - nsteps_completed, final_iters_num_triangles))
    sampler.run_mcmc(None, trial_samp_len - nsteps_completed, progress=True)

# Close multiprocessing pool
mp_pool.close()

# Print burnin and thin lengths
tau = backend.get_autocorr_time()
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))

print("Samples burn-in: {0}".format(burnin))
print("Samples thin: {0}".format(thin))


print(backend.get_chain())
print(backend.get_log_prob())
print(backend.get_log_prior())

