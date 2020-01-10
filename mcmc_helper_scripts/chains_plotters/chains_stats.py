#!/usr/bin/env python

# Chains Plotter
# ---
# Abhimat Gautam

import numpy as np

from astropy.table import Table

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.ticker import MultipleLocator

import emcee

# Read in data
trial_num = 1
filename = './chains_try{0}.h5'.format(trial_num)
reader = emcee.backends.HDFBackend(filename, read_only=True)

samples = reader.get_chain()
(num_steps, num_chains, num_params) = samples.shape

out_str = ''

out_str += "Number of Steps: {0}\n".format(num_steps)
out_str += "Number of Chains: {0}\n".format(num_chains)
out_str += "Number of Parameters: {0}\n\n".format(num_params)

log_prob_samples = reader.get_log_prob()
log_prior_samples = reader.get_blobs()

# print(log_prob_samples.shape)
# print(log_prior_samples.shape)

# Print burnin and thin lengths
tau = reader.get_autocorr_time(quiet=True)    

burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))

out_str += "Chains burn-in length: {0}\n".format(burnin)
out_str += "Chains thin length: {0}\n".format(thin)
out_str += "Tau: {0}\n".format(tau)

print(out_str)
with open('chains_stats_try1.txt', 'w') as out_file:
    out_file.write(out_str)
