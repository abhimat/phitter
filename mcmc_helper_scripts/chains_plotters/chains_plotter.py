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



# Set up plot
plt.style.use(['tex_paper', 'ticks_outtie'])

fig = plt.figure(figsize=(10,20))

parameters = [r"$A_{K'}$", r"$A_{H}$ Mod",
              r"$R_{1}$ $(R_{\odot})$", r"$R_{2}$ $(R_{\odot})$",
              r"$i$ $(\degree)$", r"$P$ (d)",
              r"$t_0$ (MJD)"]

## Set up subplot for each parameter
plot_axs = []
for param_index in range(num_params):
    cur_ax = fig.add_subplot(num_params, 1, param_index + 1)

    cur_ax.set_ylabel(parameters[param_index])

    if param_index == (num_params - 1):
        cur_ax.set_xlabel(r"Step Number")

    plot_axs.append(cur_ax)


# Plot each parameter for each chain
chain_nums = range(num_chains)
highlight_chain_nums = np.random.choice(chain_nums, size=10, replace=False)

plot_alpha=0.2
if num_chains > 200:
    plot_alpha = 0.05

for cur_chain_num in chain_nums:
    cur_chain_rows = samples[:, cur_chain_num, :]
    
    chain_color='k'
    chain_alpha=plot_alpha
    if cur_chain_num in highlight_chain_nums:
        chain_color='r'
        chain_alpha=0.5

    for param_index in range(num_params):
        cur_chain_param_samps = cur_chain_rows[:, param_index]
        
        plot_axs[param_index].plot(range(1,num_steps+1,1),
                                   cur_chain_param_samps,
                                   '-', color=chain_color, alpha=chain_alpha)
        
        plot_axs[param_index].set_xlim([0, num_steps])



# Save out plot
fig.tight_layout()
fig.savefig('./chains_try{0}.pdf'.format(trial_num))
plt.close(fig)


# Make log prob plot
fig = plt.figure(figsize=(10,4))

ax1 = fig.add_subplot(1,1,1)

ax1.set_xlabel(r"Step Number")
ax1.set_ylabel(r"log Probability")

plot_alpha=0.2
if num_chains > 200:
    plot_alpha = 0.05

for cur_chain_num in chain_nums:
    chain_color='k'
    chain_alpha=plot_alpha
    if cur_chain_num in highlight_chain_nums:
        chain_color='r'
        chain_alpha=0.5
    
    ax1.plot(range(1,num_steps+1,1),
             log_prob_samples[:, cur_chain_num],
             '-', color=chain_color, alpha=chain_alpha)

ax1.set_xlim([0, num_steps])

if num_steps < 200:
    y_majorLocator = MultipleLocator(200)
    y_minorLocator = MultipleLocator(50)
    ax1.yaxis.set_major_locator(y_majorLocator)
    ax1.yaxis.set_minor_locator(y_minorLocator)
    
elif num_steps < 500:
    ax1.set_ylim([-250, -50])
    
    y_majorLocator = MultipleLocator(20)
    y_minorLocator = MultipleLocator(5)
    ax1.yaxis.set_major_locator(y_majorLocator)
    ax1.yaxis.set_minor_locator(y_minorLocator)
    
    x_majorLocator = MultipleLocator(100)
    x_minorLocator = MultipleLocator(20)
    ax1.xaxis.set_major_locator(x_majorLocator)
    ax1.xaxis.set_minor_locator(x_minorLocator)
else:
    ax1.set_ylim([-100, -50])
    
    y_majorLocator = MultipleLocator(10)
    y_minorLocator = MultipleLocator(2)
    ax1.yaxis.set_major_locator(y_majorLocator)
    ax1.yaxis.set_minor_locator(y_minorLocator)
    
    x_majorLocator = MultipleLocator(200)
    x_minorLocator = MultipleLocator(50)
    ax1.xaxis.set_major_locator(x_majorLocator)
    ax1.xaxis.set_minor_locator(x_minorLocator)

# Save out plot
fig.tight_layout()
fig.savefig('./log_prob_try{0}.pdf'.format(trial_num))
plt.close(fig)

# # Print out statistics on parameters
# parameter_names = ['K_ext', 'H_ext_mod', 'rad_1', 'rad_2', 'inc', 'period', 't0']
#
#
# for param_index in range(len(parameter_names)):
#     param_col_name = 'col{0}'.format(param_index + 2)
#
#     param_samps = (chains_table[param_col_name])[(burn_ignore_len*len(chain_nums)):]
#
#     cur_param_percentiles = np.percentile(param_samps, [15.866, 50., 84.134])
#     cur_param_med = cur_param_percentiles[1]
#     cur_param_hi_unc = cur_param_percentiles[2] - cur_param_med
#     cur_param_lo_unc = cur_param_med - cur_param_percentiles[0]
#
#     print('{0} = {1:.5f} + {2:.5f} - {3:.5f}'.format(parameter_names[param_index], cur_param_med, cur_param_hi_unc, cur_param_lo_unc))