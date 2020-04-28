#!/usr/bin/env python

# Model Uncertainties at Observation Times Plotter
# ---
# Abhimat Gautam


# Imports
import numpy as np
import cPickle as pickle

trial_num = 1
burn_ignore_len = 500
num_plot_samples = 100


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


# Read in calculated values
with open('./obs_modelUnc_try{0}.pkl'.format(trial_num), 'rb') as input_pickle:
    plot_indices = pickle.load(input_pickle)
    plot_binary_params = pickle.load(input_pickle)
    model_good_trials = pickle.load(input_pickle)
    model_obs_trials_kp = pickle.load(input_pickle)
    model_obs_trials_h = pickle.load(input_pickle)


## Check for trials where model didn't work
good_trials = np.where(model_good_trials == 1.)


# Kp
model_kp_medians = np.zeros(len(kp_target_MJDs))
model_kp_sig1_lo = np.zeros(len(kp_target_MJDs))
model_kp_sig1_hi = np.zeros(len(kp_target_MJDs))
model_kp_sig2_lo = np.zeros(len(kp_target_MJDs))
model_kp_sig2_hi = np.zeros(len(kp_target_MJDs))
model_kp_sig3_lo = np.zeros(len(kp_target_MJDs))
model_kp_sig3_hi = np.zeros(len(kp_target_MJDs))

for cur_kp_obs_index in range(len(kp_target_MJDs)):
    cur_obs_trials = (model_obs_trials_kp[good_trials])[:, cur_kp_obs_index]

    [model_kp_sig3_lo[cur_kp_obs_index],
     model_kp_sig2_lo[cur_kp_obs_index],
     model_kp_sig1_lo[cur_kp_obs_index],
     model_kp_medians[cur_kp_obs_index],
     model_kp_sig1_hi[cur_kp_obs_index],
     model_kp_sig2_hi[cur_kp_obs_index],
     model_kp_sig3_hi[cur_kp_obs_index]] = np.percentile(cur_obs_trials, [0.135, 2.275, 15.866,
                                               50., 84.134, 97.725, 99.865])

## H
model_h_medians = np.zeros(len(h_target_MJDs))
model_h_sig1_lo = np.zeros(len(h_target_MJDs))
model_h_sig1_hi = np.zeros(len(h_target_MJDs))
model_h_sig2_lo = np.zeros(len(h_target_MJDs))
model_h_sig2_hi = np.zeros(len(h_target_MJDs))
model_h_sig3_lo = np.zeros(len(h_target_MJDs))
model_h_sig3_hi = np.zeros(len(h_target_MJDs))

for cur_h_obs_index in range(len(h_target_MJDs)):
    cur_obs_trials = (model_obs_trials_h[good_trials])[:, cur_h_obs_index]

    [model_h_sig3_lo[cur_h_obs_index],
     model_h_sig2_lo[cur_h_obs_index],
     model_h_sig1_lo[cur_h_obs_index],
     model_h_medians[cur_h_obs_index],
     model_h_sig1_hi[cur_h_obs_index],
     model_h_sig2_hi[cur_h_obs_index],
     model_h_sig3_hi[cur_h_obs_index]] = np.percentile(cur_obs_trials, [0.135, 2.275, 15.866,
                                             50., 84.134, 97.725, 99.865])











# Get astropy time objects from MJDs
from astropy.time import Time

kp_times = Time(kp_target_MJDs, format='mjd')
h_times = Time(h_target_MJDs, format='mjd')




# Make plot
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.ticker import MultipleLocator

#### Plot Nerdery
plt.rc('font', family='serif')
plt.rc('font', serif='Computer Modern Roman')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r"\usepackage{gensymb}")

plt.rc('xtick', direction = 'in')
plt.rc('ytick', direction = 'in')
plt.rc('xtick', top = True)
plt.rc('ytick', right = True)

fig = plt.figure(figsize=(8,6))
y_lim_extents = 0.7

## Kp Light Curve
ax1 = fig.add_subplot(2, 1, 1)

# ax1.set_xlabel(r"")
ax1.set_ylabel(r"$m_{K'}$")
ax1.invert_yaxis()

# ### Model uncertainty regions
# for cur_kp_obs_index in range(len(kp_target_MJDs)):
#     cur_obs_time = (kp_times[cur_kp_obs_index]).jyear
#
#     ax1.plot([cur_obs_time, cur_obs_time], [model_kp_sig3_lo[cur_kp_obs_index], model_kp_sig3_hi[cur_kp_obs_index]], '-', color='r', lw=4., alpha=0.2)
#     ax1.plot([cur_obs_time, cur_obs_time], [model_kp_sig2_lo[cur_kp_obs_index], model_kp_sig2_hi[cur_kp_obs_index]], '-', color='r', lw=4., alpha=0.2)
#     ax1.plot([cur_obs_time, cur_obs_time], [model_kp_sig1_lo[cur_kp_obs_index], model_kp_sig1_hi[cur_kp_obs_index]], '-', color='r', lw=4., alpha=0.2)


#### Data Points
ax1.errorbar(kp_times.jyear, kp_target_mags, yerr = kp_target_mag_errors, fmt='k.', alpha = 0.8)

#### Violin plots for model values
vp_parts = ax1.violinplot(model_obs_trials_kp[good_trials], kp_times.jyear,
                          widths = 0.25, vert=True, showextrema=False)

for pc in vp_parts['bodies']:
    pc.set_facecolor('r')
    pc.set_edgecolor('r')
    # pc.set_alpha(1)

ax1.set_xlim([2006, 2020])
ax1.set_ylim([np.mean(kp_target_mags) + y_lim_extents/2., np.mean(kp_target_mags) - y_lim_extents/2.])

x_majorLocator = MultipleLocator(2.)
x_minorLocator = MultipleLocator(0.5)
ax1.xaxis.set_major_locator(x_majorLocator)
ax1.xaxis.set_minor_locator(x_minorLocator)

y_majorLocator = MultipleLocator(0.2)
y_minorLocator = MultipleLocator(0.05)
ax1.yaxis.set_major_locator(y_majorLocator)
ax1.yaxis.set_minor_locator(y_minorLocator)


## H Light Curve
ax2 = fig.add_subplot(2, 1, 2)

ax2.set_xlabel(r"Observation Time")
ax2.set_ylabel(r"$m_{H}$")
ax2.invert_yaxis()

# ### Model uncertainty regions
# for cur_h_obs_index in range(len(h_target_MJDs)):
#     cur_obs_time = (h_times[cur_h_obs_index]).jyear
#
#     ax2.plot([cur_obs_time, cur_obs_time], [model_h_sig3_lo[cur_h_obs_index], model_h_sig3_hi[cur_h_obs_index]], '-', color='r', lw=4., alpha=0.2)
#     ax2.plot([cur_obs_time, cur_obs_time], [model_h_sig2_lo[cur_h_obs_index], model_h_sig2_hi[cur_h_obs_index]], '-', color='r', lw=4., alpha=0.2)
#     ax2.plot([cur_obs_time, cur_obs_time], [model_h_sig1_lo[cur_h_obs_index], model_h_sig1_hi[cur_h_obs_index]], '-', color='r', lw=4., alpha=0.2)


#### Data Points
ax2.errorbar(h_times.jyear, h_target_mags, yerr = h_target_mag_errors, fmt='k.', alpha = 0.8)

#### Violin plots for model values
vp_parts = ax2.violinplot(model_obs_trials_h[good_trials], h_times.jyear,
                          widths = 0.25, vert=True, showextrema=False)

for pc in vp_parts['bodies']:
    pc.set_facecolor('royalblue')
    pc.set_edgecolor('royalblue')
    # pc.set_alpha(1)

ax2.set_xlim([2006, 2020])
ax2.set_ylim([np.mean(h_target_mags) + y_lim_extents/2., np.mean(h_target_mags) - y_lim_extents/2.])

x_majorLocator = MultipleLocator(2.)
x_minorLocator = MultipleLocator(0.5)
ax2.xaxis.set_major_locator(x_majorLocator)
ax2.xaxis.set_minor_locator(x_minorLocator)

y_majorLocator = MultipleLocator(0.2)
y_minorLocator = MultipleLocator(0.05)
ax2.yaxis.set_major_locator(y_majorLocator)
ax2.yaxis.set_minor_locator(y_minorLocator)


fig.tight_layout()
fig.savefig('./obs_modelUnc_try{0}.pdf'.format(trial_num))
plt.close(fig)
