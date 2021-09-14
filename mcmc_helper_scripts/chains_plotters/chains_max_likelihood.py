#!/usr/bin/env python

# Chains Max Likelihood
# ---
# Abhimat Gautam

import numpy as np

from astropy.table import Table

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.ticker import MultipleLocator

import emcee

# Fit parameters
params = ['Kp_ext', 'H_ext_mod',
          'star1_rad', 'star2_rad',
          'bin_inc', 'bin_period',
          't0']
param_units = ['', '',
               'R_sun', 'R_sun',
               'degrees', 'days',
               '']


# Read in data
trial_num = 1
filename = './chains_try{0}.h5'.format(trial_num)
reader = emcee.backends.HDFBackend(filename, read_only=True)

samples = reader.get_chain()
(num_steps, num_chains, num_params) = samples.shape

out_str = ''

log_prob_samples = reader.get_log_prob()
log_prior_samples = reader.get_blobs()

# print(log_prob_samples)
# print(log_prior_samples)

log_prob_max = np.max(log_prob_samples)
log_prob_argmax_index = np.unravel_index(np.argmax(log_prob_samples), log_prob_samples.shape)

dof = 69-7
red_chi_sq_min = -2. * log_prob_max / dof


out_str += "Log Likelihood max: {0}\n".format(log_prob_max)
out_str += "Reduced Chi Squared min: {0}\n".format(red_chi_sq_min)
out_str += "Log Likelihood argmax: {0}\n\n".format(log_prob_argmax_index)

out_str += "Parameter values at argmax\n"

for param_index in range(len(params)):
    cur_param_samples = samples[:, :, param_index]
    
    cur_param_argmax_val = cur_param_samples[log_prob_argmax_index]
    
    out_str += "{0} = {1} {2}\n".format(params[param_index], cur_param_argmax_val, param_units[param_index])

print(out_str)
with open('chains_max_likelihood_try1.txt', 'w') as out_file:
    out_file.write(out_str)
