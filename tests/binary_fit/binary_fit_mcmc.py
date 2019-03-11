#!/usr/bin/env python

# Binary light curve fitter
# Using PopStar with MIST tracks and PHOEBE binary models
# ---
# Abhimat Gautam

import numpy as np
from scipy.optimize import leastsq

from phoebe_phitter import mcmc_fit

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.ticker import MultipleLocator

import cPickle as pickle
import sys
from tqdm import tqdm


# trial_num = 0
# trial_num_walkers = 16
# trial_burn_len = 1
# trial_samp_len = 10

trial_num = 1
trial_num_walkers = 100
trial_burn_len = 10
trial_samp_len = 1100

# Isochrone parameters
isoc_age = 7.5e9
isoc_ext = 2.63
isoc_dist = 7.971e3
isoc_phase = 'RGB'

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
init_Kp_ext_t = 2.33
init_H_ext_mod_t = 0.0
init_bin_inc_t = 77.6
init_star1_rad_t = 37.5
init_star2_rad_t = 15.8
init_binary_period_t = 78.8
init_binary_ecc_t = -0.0054
init_t0_t = 53778.93

# Set up MCMC fitting object
mcmc_fit_obj = mcmc_fit.mcmc_fitter_rad_interp()

## Make isochrone that will be interpolated
mcmc_fit_obj.make_isochrone(isoc_age, isoc_ext, isoc_dist, isoc_phase)

## Set observation times used during fitting (in MJDs)
mcmc_fit_obj.set_observation_times(kp_target_MJDs, h_target_MJDs)

## Set observation mags, to compare with model mags
mcmc_fit_obj.set_observation_mags(
    kp_target_mags, kp_target_mag_errors,
    h_target_mags, h_target_mag_errors)


# Set up Emcee
def emcee_lnprob(theta):
    return mcmc_fit_obj.lnprob(theta)


ndim, nwalkers = 8, trial_num_walkers
p0 = [(init_Kp_ext_t + 1e-1*(np.random.rand() - 0.5),
       init_H_ext_mod_t + 1e-1*(np.random.rand() - 0.5),
       init_star1_rad_t + 1e0*(np.random.rand() - 0.5),
       init_star2_rad_t + 1e0*(np.random.rand() - 0.5),
       init_bin_inc_t + 1e1*(np.random.rand() - 0.5),
       init_binary_period_t + 1e-1*(np.random.rand() - 0.5),
       init_binary_ecc_t + 1e-2*(np.random.rand() - 0.5),
       init_t0_t + 1e0*(np.random.rand() - 0.5)) for i in range(nwalkers)]

test_theta = (init_Kp_ext_t, init_H_ext_mod_t,
              init_star1_rad_t, init_star2_rad_t,
              init_bin_inc_t, init_binary_period_t,
              init_binary_ecc_t, init_t0_t)

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
sampler = emcee.EnsembleSampler(nwalkers, ndim, emcee_lnprob, threads=7)   # , threads=16

## Burn-in
burn_len = trial_burn_len
pos, prob, state = sampler.run_mcmc(p0, burn_len)
sampler.reset()

## Running sampler, after burn-in
nsteps = trial_samp_len

f = open("./chains/chains_try{0}.dat".format(trial_num), "w")
f.close()

# sampler.run_mcmc(pos, nsteps)
for i, result in tqdm(enumerate(sampler.sample(pos, iterations=nsteps))):
    position = result[0]
    
    f = open("./chains/chains_try{0}.dat".format(trial_num), "a")
    for k in range(position.shape[0]):
        f.write("{0:4d} {1:s}\n".format(k, " ".join(map(str, position[k].tolist()))))
    f.close()
    
    # if (i+1) % 100 == 0:
    #     print("Progress: {0:5.1%}".format(float(i) / nsteps))


print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

samples = sampler.flatchain

## Write out samples to pickle
import cPickle as pickle
with open('./emcee_samples_try{0}.pkl'.format(trial_num), 'wb') as output_pickle:
    pickle.dump(samples, output_pickle)

samples_Kp_ext = samples[:, 0]
samples_H_ext_mod = samples[:, 1]
samples_bin_inc = samples[:, 2]
samples_star1_rad = samples[:, 3]
samples_star2_rad = samples[:, 4]
samples_binary_period = samples[:, 5]
samples_binary_ecc = samples[:, 6]
samples_t0 = samples[:, 7]

## Print out MCMC results
### Compute median centered 1 sigma regions
Kp_ext_mcmc, H_ext_mod_mcmc, bin_inc_mcmc, star1_rad_mcmc, star2_rad_mcmc, binary_period_mcmc, binary_ecc_mcmc, t0_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [15.866, 50., 84.134], axis=0)))

print('Kp_ext = {0:.4f} + {1:.4f} - {2:.4f}'.format(Kp_ext_mcmc[0], Kp_ext_mcmc[1], Kp_ext_mcmc[2]))
print('H_ext_mod = {0:.4f} + {1:.4f} - {2:.4f}'.format(H_ext_mod_mcmc[0], H_ext_mod_mcmc[1], H_ext_mod_mcmc[2]))
print('bin_inc = {0:.4f} + {1:.4f} - {2:.4f}'.format(bin_inc_mcmc[0], bin_inc_mcmc[1], bin_inc_mcmc[2]))
print('star1_rad = {0:.4f} + {1:.4f} - {2:.4f}'.format(star1_rad_mcmc[0], star1_rad_mcmc[1], star1_rad_mcmc[2]))
print('star2_rad = {0:.4f} + {1:.4f} - {2:.4f}'.format(star2_rad_mcmc[0], star2_rad_mcmc[1], star2_rad_mcmc[2]))
print('binary_period = {0:.4f} + {1:.4f} - {2:.4f}'.format(binary_period_mcmc[0], binary_period_mcmc[1], binary_period_mcmc[2]))
print('binary_ecc = {0:.4f} + {1:.4f} - {2:.4f}'.format(binary_ecc_mcmc[0], binary_ecc_mcmc[1], binary_ecc_mcmc[2]))
print('t0 = {0:.4f} + {1:.4f} - {2:.4f}'.format(t0_mcmc[0], t0_mcmc[1], t0_mcmc[2]))
