#!/usr/bin/env python

# Model Uncertainties at Observation Times Calculator
# ---
# Abhimat Gautam


# Imports
import numpy as np

from phoebe import u
from phoebe import c as const

from phoebe_phitter import lc_calc, mcmc_fit

from multiprocessing import Pool
import parmap
parallel_cores = 7

import cPickle as pickle
import time

trial_num = 1
burn_ignore_len = 500
num_plot_samples = 200


# Isochrone parameters
isoc_age = 12.8e9
isoc_ext = 2.63
isoc_dist = 7.971e3
isoc_phase = 'RGB'
isoc_met = 0.5


early_iters_cutoff = 200
early_iters_num_triangles = 200
final_iters_num_triangles = 500


# Read in observation data
target_star = 'S2-36'
with open('../lc_data.pkl', 'rb') as input_pickle:
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

# Set up MCMC fitting object
## We'll be regenerating model light curves using this MCMC object

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
mcmc_fit_obj.set_model_numTriangles(final_iters_num_triangles)

## Set to use blackbody atmosphere
mcmc_fit_obj.set_model_use_blackbody_atm(True)

## Set to model H extinction modifier
mcmc_fit_obj.set_model_H_ext_mod(True)

## Set to not model eccentricity
mcmc_fit_obj.set_model_eccentricity(False)

## Set to model distance
mcmc_fit_obj.set_model_distance(False)
mcmc_fit_obj.default_dist = 7.971e3


# Extract random sets of parameters from the chains

# test_theta = (init_Kp_ext_t, init_H_ext_mod_t,
#               init_star1_rad_t, init_star2_rad_t,
#               init_binary_inc_t, init_binary_period_t,
#               init_t0_t)

## Chains file name
filename = '../chains/chains_try{0}.h5'.format(trial_num)

## Read in sample
import emcee
reader = emcee.backends.HDFBackend(filename, read_only=True)

samples = reader.get_chain()
(num_steps, num_chains, num_params) = samples.shape
samples = reader.get_chain(flat=True)

## Random indices for calculation
total_samples = (num_steps - burn_ignore_len) * num_chains

plot_indices = np.random.choice(total_samples, size=num_plot_samples, replace=False)

## Samples at random indices
samples = samples[burn_ignore_len * num_chains:,:]

plot_binary_params = samples[plot_indices]



# Generate binary light curves at the random parameter indices

## Presorted indices
kp_obs_indices = np.array(range(len(kp_target_MJDs)))
h_obs_indices = np.array(range(len(h_target_MJDs)))

## Function for parallelization
def binary_lc_run(run_num, binary_params):
    cur_binary_params = binary_params[run_num]
    cur_theta = (cur_binary_params[0], cur_binary_params[1],
                 cur_binary_params[2], cur_binary_params[3],
                 cur_binary_params[4], cur_binary_params[5],
                 cur_binary_params[6])
    
    (cur_model_mags_Kp, cur_model_mags_H) = mcmc_fit_obj.calculate_model_lc(cur_theta)
    
    ### Sort back modeled data points into original order 
    binary_period_t = cur_binary_params[5]
    t0_t = cur_binary_params[6]
    
    (kp_phase_out, h_phase_out) = lc_calc.phased_obs(
                                      (kp_target_MJDs, h_target_MJDs),
                                      binary_period_t * u.d, t0_t)
    
    (kp_phased_days, kp_phases_sorted_inds, kp_model_times) = kp_phase_out
    (h_phased_days, h_phases_sorted_inds, h_model_times) = h_phase_out
    
    #### Indices for sorting back into original order
    kp_obs_indices_sorted = kp_obs_indices[kp_phases_sorted_inds]
    h_obs_indices_sorted = h_obs_indices[h_phases_sorted_inds]
    
    kp_model_to_obs_sorted_inds = np.argsort(kp_obs_indices_sorted)
    h_model_to_obs_sorted_inds = np.argsort(h_obs_indices_sorted)
    
    cur_model_mags_Kp = cur_model_mags_Kp[kp_model_to_obs_sorted_inds]
    cur_model_mags_H = cur_model_mags_H[h_model_to_obs_sorted_inds]
    
    return [cur_model_mags_Kp, cur_model_mags_H]


## Calculate model light curves with parallelization
start_time = time.time()
binary_lc_run_pool = Pool(processes=parallel_cores)
binary_lc_result = parmap.map(binary_lc_run,
                              range(num_plot_samples), plot_binary_params,
                              pool=binary_lc_run_pool)
end_time = time.time()

print('Number of sample binary models = {0}'.format(num_plot_samples))
print('Total binary modeling time = {0:.3f} sec'.format(end_time - start_time))

## Re-shape pool outputs into numpy arrays
model_good_trials = np.zeros(num_plot_samples)
model_obs_trials_kp = np.zeros((num_plot_samples, len(kp_target_MJDs)))
model_obs_trials_h = np.zeros((num_plot_samples, len(h_target_MJDs)))

for samp_run in range(num_plot_samples):
    [cur_model_mags_Kp, cur_model_mags_H] = binary_lc_result[samp_run]
    
    if (cur_model_mags_Kp[0] == -1.) or (cur_model_mags_H[0] == -1.):
        model_good_trials[samp_run] = 0
        continue
    else:
        model_good_trials[samp_run] = 1
        model_obs_trials_kp[samp_run] = cur_model_mags_Kp
        model_obs_trials_h[samp_run] = cur_model_mags_H



# Save out calculated values
with open('./obs_modelUnc_try{0}.pkl'.format(trial_num), 'wb') as output_pickle:
    pickle.dump(plot_indices, output_pickle)
    pickle.dump(plot_binary_params, output_pickle)
    pickle.dump(model_good_trials, output_pickle)
    pickle.dump(model_obs_trials_kp, output_pickle)
    pickle.dump(model_obs_trials_h, output_pickle)
