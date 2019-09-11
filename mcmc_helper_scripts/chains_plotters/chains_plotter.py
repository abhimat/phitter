#!/usr/bin/env python

# Chains Plotter
# ---
# Abhimat Gautam

import numpy as np

from astropy.table import Table

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.ticker import MultipleLocator

# Set up plot
#### Plot Nerdery
plt.rc('font', family='serif')
plt.rc('font', serif='Computer Modern Roman')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r"\usepackage{gensymb}")

plt.rc('xtick', direction = 'in')
plt.rc('ytick', direction = 'in')
plt.rc('xtick', top = True)
plt.rc('ytick', right = True)

fig = plt.figure(figsize=(10,20))

parameters = [r"$A_{K'}$",
              r"$R_{1}$ $(R_{\odot})$", r"$R_{2}$ $(R_{\odot})$",
              r"$i$ $(\degree)$", r"$P$ (d)", r"$d$ (pc)",
              r"$t_0$ (MJD)"]
num_params = len(parameters)

## Set up subplot for each parameter
plot_axs = []
for param_index in range(num_params):
    cur_ax = fig.add_subplot(num_params, 1, param_index + 1)
    
    cur_ax.set_ylabel(parameters[param_index])
    
    if param_index == (num_params - 1):
        cur_ax.set_xlabel(r"Iteration")
    
    plot_axs.append(cur_ax)

# Read in data
trial_num = 0
trial_num = 1

chains_table = Table.read('./chains_try{0}.dat'.format(trial_num), format='ascii')

burn_ignore_len = 0
# burn_ignore_len = 100

# Plot each parameter for each chain
chain_nums = np.unique(chains_table['col1'])
highlight_chain_nums = np.random.choice(chain_nums, size=10, replace=False)

plot_alpha=0.2
if len(chain_nums) > 200:
    plot_alpha = 0.05

for cur_chain_num in chain_nums:
    cur_chain_rows = chains_table[np.where(chains_table['col1'] == cur_chain_num)]
    
    chain_color='k'
    chain_alpha=plot_alpha
    if cur_chain_num in highlight_chain_nums:
        chain_color='r'
        chain_alpha=0.5
    
    for param_index in range(num_params):
        param_col_name = 'col{0}'.format(param_index + 2)
        
        cur_chain_param_samps = (cur_chain_rows[param_col_name])[burn_ignore_len:]
        
        if cur_chain_num == 0 and param_index == 0:
            print('Number of samples/walker: {0}'.format(len(cur_chain_param_samps)))
        
        plot_axs[param_index].plot(range(1,len(cur_chain_param_samps)+1,1),
                                   cur_chain_param_samps,
                                   '-', color=chain_color, alpha=chain_alpha)



# Save out plot
fig.tight_layout()
fig.savefig('./chains_try{0}.pdf'.format(trial_num))
plt.close(fig)



# Print out statistics on parameters
parameter_names = ['K_ext', 'rad_1', 'rad_2', 'inc', 'period', 'dist', 't0']


for param_index in range(len(parameter_names)):
    param_col_name = 'col{0}'.format(param_index + 2)
    
    param_samps = (chains_table[param_col_name])[(burn_ignore_len*len(chain_nums)):]
    
    cur_param_percentiles = np.percentile(param_samps, [15.866, 50., 84.134])
    cur_param_med = cur_param_percentiles[1]
    cur_param_hi_unc = cur_param_percentiles[2] - cur_param_med
    cur_param_lo_unc = cur_param_med - cur_param_percentiles[0]
    
    print('{0} = {1:.5f} + {2:.5f} - {3:.5f}'.format(parameter_names[param_index], cur_param_med, cur_param_hi_unc, cur_param_lo_unc))