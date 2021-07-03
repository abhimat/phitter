#!/usr/bin/env python

# Chains Corner Plotter
# ---
# Abhimat Gautam

import numpy as np

from astropy.table import Table

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.ticker import MultipleLocator

parameters = [r"$A_{K'}$", r"$A_{H}$ Mod", r"$R_{1}$ $(R_{\odot})$", r"$R_{2}$ $(R_{\odot})$", r"$i$ $(\degree)$", r"$P$ (d)", r"$t_0$ (MJD)"]

import emcee

# Read in data
trial_num = 1
filename = './chains_try{0}.h5'.format(trial_num)
reader = emcee.backends.HDFBackend(filename, read_only=True)

samples = reader.get_chain()
(num_steps, num_chains, num_params) = samples.shape
samples = reader.get_chain(flat=True)
# print(samples.shape)

print("Number of Steps: {0}".format(num_steps))
print("Number of Chains: {0}".format(num_chains))
print("Number of Parameters: {0}".format(num_params))

log_prob_samples = reader.get_log_prob(flat=True)
log_prior_samples = reader.get_blobs(flat=True)

# print(log_prob_samples.shape)
# print(log_prior_samples.shape)


# Construct samples array for plotting

samples_orig = np.copy(samples)

burn_ignore_len = 0
burn_ignore_len = 500
last_steps_count = -1

## Organize into shape that can work for us
if last_steps_count == -1:
    print('---')
    print('Ignoring first {0} steps'.format(burn_ignore_len))
    samples = samples[burn_ignore_len * num_chains:,:]
else:
    print('---')
    print('Plotting last {0} steps'.format(last_steps_count))
    samples = samples[(-1 * last_steps_count * num_chains):,:]
# print(samples)
# samples = np.swapaxes(samples, 0, 1)


samples_Kp_ext = samples[:, 0]
samples_H_ext_mod = samples[:, 1]
samples_star1_rad = samples[:, 2]
samples_star2_rad = samples[:, 3]
samples_bin_inc = samples[:, 4]
samples_binary_period = samples[:, 5]
# samples_binary_dist = samples[:, 5]
samples_t0 = samples[:, 6]

## Print out MCMC results
### Compute median centered 1 sigma regions
Kp_ext_mcmc, H_ext_mod_mcmc, star1_rad_mcmc, star2_rad_mcmc, bin_inc_mcmc, binary_period_mcmc, t0_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [15.866, 50., 84.134], axis=0)))

print('---')
print('Kp_ext = {0:.4f} + {1:.4f} - {2:.4f}'.format(Kp_ext_mcmc[0], Kp_ext_mcmc[1], Kp_ext_mcmc[2]))
print('H_ext_mod = {0:.4f} + {1:.4f} - {2:.4f}'.format(H_ext_mod_mcmc[0], H_ext_mod_mcmc[1], H_ext_mod_mcmc[2]))
print('star1_rad = {0:.4f} + {1:.4f} - {2:.4f}'.format(star1_rad_mcmc[0], star1_rad_mcmc[1], star1_rad_mcmc[2]))
print('star2_rad = {0:.4f} + {1:.4f} - {2:.4f}'.format(star2_rad_mcmc[0], star2_rad_mcmc[1], star2_rad_mcmc[2]))
print('bin_inc = {0:.4f} + {1:.4f} - {2:.4f}'.format(bin_inc_mcmc[0], bin_inc_mcmc[1], bin_inc_mcmc[2]))
print('binary_period = {0:.5f} + {1:.5f} - {2:.5f}'.format(binary_period_mcmc[0], binary_period_mcmc[1], binary_period_mcmc[2]))
# print('binary_dist = {0:.4f} + {1:.4f} - {2:.4f}'.format(binary_dist_mcmc[0], binary_dist_mcmc[1], binary_dist_mcmc[2]))
print('t0 = {0:.4f} + {1:.4f} - {2:.4f}'.format(t0_mcmc[0], t0_mcmc[1], t0_mcmc[2]))


## Corner Plot
plt.style.use(['tex_paper', 'ticks_outtie'])

import corner
fig = corner.corner(samples, labels=parameters)
fig.tight_layout()
fig.savefig("chains_corner_try{0}.pdf".format(trial_num))