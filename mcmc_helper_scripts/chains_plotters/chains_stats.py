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

print("Number of Steps: {0}".format(num_steps))
print("Number of Chains: {0}".format(num_chains))
print("Number of Parameters: {0}".format(num_params))

log_prob_samples = reader.get_log_prob()
log_prior_samples = reader.get_blobs()

# print(log_prob_samples.shape)
# print(log_prior_samples.shape)

# Print burnin and thin lengths
try:
    tau = reader.get_autocorr_time()
except emcee.autocorr.AutocorrError as e:
    print('Too short chains :(')
    print(e)
    

burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))

print("Chains burn-in length: {0}".format(burnin))
print("Chains thin length: {0}".format(thin))
print("Tau: {0}".format(tau))