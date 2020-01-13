#!/usr/bin/env python

# Binary light curve plotter
# ---
# Abhimat Gautam

from phoebe_phitter import isoc_interp, lc_calc

from phoebe import u
from phoebe import c as const

import numpy as np

import cPickle as pickle

# Isochrone parameters
isoc_age = 13.5e9
isoc_ext = 2.63
isoc_dist = 7.971e3
isoc_phase = 'RGB'
isoc_met = 0.0

# Filter properties
lambda_Ks = 2.18e-6 * u.m
dlambda_Ks = 0.35e-6 * u.m

lambda_Kp = 2.124e-6 * u.m
dlambda_Kp = 0.351e-6 * u.m

lambda_H = 1.633e-6 * u.m
dlambda_H = 0.296e-6 * u.m

# Extinction law (using Nogueras-Lara+ 2018)
ext_alpha = 2.30

# Distance to GC
## From Do+ 2019 (GR2018 paper)
gc_dist = 7.971e3 * u.pc

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


# Read in chains to get maximum likelihood
import emcee
trial_num = 1
filename = '../chains/chains_try{0}.h5'.format(trial_num)
reader = emcee.backends.HDFBackend(filename, read_only=True)

samples = reader.get_chain()
(num_steps, num_chains, num_params) = samples.shape

log_prob_samples = reader.get_log_prob()
log_prior_samples = reader.get_blobs()

log_prob_max = np.max(log_prob_samples)
log_prob_argmax_index = np.unravel_index(np.argmax(log_prob_samples), log_prob_samples.shape)

# Model parameters
Kp_ext_t = (samples[:, :, 0])[log_prob_argmax_index]
H_ext_mod_t = (samples[:, :, 1])[log_prob_argmax_index]
star1_rad_t = (samples[:, :, 2])[log_prob_argmax_index]
star2_rad_t = (samples[:, :, 3])[log_prob_argmax_index]
binary_inc_t = (samples[:, :, 4])[log_prob_argmax_index]
binary_period_t = (samples[:, :, 5])[log_prob_argmax_index]
binary_ecc_t = 0.0
binary_dist_t = 7971
t0_t = (samples[:, :, 6])[log_prob_argmax_index]

H_ext_t = Kp_ext_t * (lambda_Kp / lambda_H)**ext_alpha + H_ext_mod_t

fit_theta = (Kp_ext_t, H_ext_mod_t,
             star1_rad_t, star2_rad_t,
             binary_inc_t, binary_period_t,
             binary_ecc_t, t0_t)

# Binary parameters
binary_inc = binary_inc_t * u.deg
binary_period = binary_period_t * u.d
binary_ecc = binary_ecc_t
t0 = t0_t

binary_params_model = (binary_period, binary_ecc, binary_inc, binary_period_t)
binary_params = (binary_period, binary_ecc, binary_inc, t0)

# Interpolate stellar parameters
isochrone = isoc_interp.isochrone_mist(age=isoc_age,
                                       ext=isoc_ext,
                                       dist=isoc_dist,
                                       phase=isoc_phase,
                                       met=isoc_met,
                                       use_atm_func='phoenix')

(star1_params_all, star1_params_lcfit) = isochrone.rad_interp(star1_rad_t)
(star2_params_all, star2_params_lcfit) = isochrone.rad_interp(star2_rad_t)

print(star1_params_all)
print(star2_params_all)


# Model phase spacing
model_phase_spacing = 0.01
model_phases = np.arange(0.0, 1.0, model_phase_spacing)

model_times = (model_phases) * binary_period_t
model_observation_times = (model_times, model_times)


# Obtain model magnitudes
num_triangles = 500
(binary_model_mags_Kp, binary_model_mags_H) = lc_calc.binary_mags_calc(
                                                  star1_params_lcfit,
                                                  star2_params_lcfit,
                                                  binary_params_model,
                                                  model_observation_times,
                                                  isoc_ext,
                                                  Kp_ext_t, H_ext_t,
                                                  ext_alpha,
                                                  gc_dist, binary_dist_t,
                                                  use_blackbody_atm=True,
                                                  make_mesh_plots=True,
                                                  plot_name='binary_mesh',
                                                  num_triangles=num_triangles)

## Repeat in phase
model_phases_plot = np.append(model_phases - 1., model_phases)
model_phases_plot = np.append(model_phases_plot, model_phases + 1.)

binary_model_mags_Kp_plot = np.append(binary_model_mags_Kp, binary_model_mags_Kp)
binary_model_mags_Kp_plot = np.append(binary_model_mags_Kp_plot, binary_model_mags_Kp)

binary_model_mags_H_plot = np.append(binary_model_mags_H, binary_model_mags_H)
binary_model_mags_H_plot = np.append(binary_model_mags_H_plot, binary_model_mags_H)


## Phase the observations
observation_times = (kp_target_MJDs, h_target_MJDs)
(kp_phase_out, h_phase_out) = lc_calc.phased_obs(
                                  observation_times,
                                  binary_period, t0)

(kp_phased_days, kp_phases_sorted_inds, kp_model_times) = kp_phase_out
(h_phased_days, h_phases_sorted_inds, h_model_times) = h_phase_out

(kp_phased_days, kp_phases_sorted_inds, kp_model_times) = kp_phase_out
(h_phased_days, h_phases_sorted_inds, h_model_times) = h_phase_out



# Plot
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

#### Phased Kp Light Curve
ax1 = fig.add_subplot(2, 1, 1)

# ax1.set_xlabel(r"Phase (Period: {0:.4f} days)".format(binary_period_t))
ax1.set_ylabel(r"$m_{K'}$")
ax1.invert_yaxis()

#### Model
ax1.plot(model_phases_plot, binary_model_mags_Kp_plot, '-', color='r', lw=2., alpha=0.4)

# #### Mean Mag
# ax1.fill_between([-0.5, 1.5], kp_mean_mag_fit - kp_mean_mag_fit_unc, y2=kp_mean_mag_fit + kp_mean_mag_fit_unc, color='royalblue', alpha=0.4)
# ax1.axhline(y=kp_mean_mag_fit, color='royalblue', alpha=0.8)

#### Data Points
highlight_indexes = []

# ax1.plot(np.sort(kp_phased_days), model_mags_Kp_dataTimes, '.', color='red', alpha=0.2)

ax1.errorbar(kp_phased_days + 1., kp_target_mags, yerr = kp_target_mag_errors, fmt='.', color='grey', alpha=0.6)
ax1.errorbar(kp_phased_days - 1., kp_target_mags, yerr = kp_target_mag_errors, fmt='.', color='grey', alpha=0.6)
ax1.errorbar(kp_phased_days, kp_target_mags, yerr = kp_target_mag_errors, fmt='k.')

ax1.errorbar(kp_phased_days[highlight_indexes], kp_target_mags[highlight_indexes], yerr = kp_target_mag_errors[highlight_indexes], fmt='.', color='royalblue')

ax1.set_xlim([-0.5, 1.5])
# ax1.set_ylim([np.mean(kp_target_mags) + y_lim_extents/2., np.mean(kp_target_mags) - y_lim_extents/2.])

x_majorLocator = MultipleLocator(0.5)
x_minorLocator = MultipleLocator(0.1)
ax1.xaxis.set_major_locator(x_majorLocator)
ax1.xaxis.set_minor_locator(x_minorLocator)

# y_majorLocator = MultipleLocator(0.2)
# y_minorLocator = MultipleLocator(0.05)
# ax1.yaxis.set_major_locator(y_majorLocator)
# ax1.yaxis.set_minor_locator(y_minorLocator)


#### Phased H Light Curve
ax2 = fig.add_subplot(2, 1, 2)

ax2.set_xlabel(r"Phase (Period: {0:.4f} days)".format(binary_period_t))
ax2.set_ylabel(r"$m_{H}$")
ax2.invert_yaxis()

#### Model
ax2.plot(model_phases_plot, binary_model_mags_H_plot, '-', color='r', lw=2., alpha=0.4)

# #### Mean Mag
# ax2.fill_between([-0.5, 1.5], kp_mean_mag_fit - kp_mean_mag_fit_unc, y2=kp_mean_mag_fit + kp_mean_mag_fit_unc, color='royalblue', alpha=0.4)
# ax2.axhline(y=kp_mean_mag_fit, color='royalblue', alpha=0.8)

#### Data Points
highlight_indexes = []

# ax2.plot(np.sort(h_phased_days), model_mags_H_dataTimes, '.', color='red', alpha=0.2)

ax2.errorbar(h_phased_days + 1., h_target_mags, yerr = h_target_mag_errors, fmt='.', color='grey', alpha=0.6)
ax2.errorbar(h_phased_days - 1., h_target_mags, yerr = h_target_mag_errors, fmt='.', color='grey', alpha=0.6)
ax2.errorbar(h_phased_days, h_target_mags, yerr = h_target_mag_errors, fmt='k.')

ax2.errorbar(h_phased_days[highlight_indexes], h_target_mags[highlight_indexes], yerr = h_target_mag_errors[highlight_indexes], fmt='.', color='royalblue')

ax2.set_xlim([-0.5, 1.5])
# ax2.set_ylim([np.mean(h_target_mags) + y_lim_extents/2., np.mean(h_target_mags) - y_lim_extents/2.])

x_majorLocator = MultipleLocator(0.5)
x_minorLocator = MultipleLocator(0.1)
ax2.xaxis.set_major_locator(x_majorLocator)
ax2.xaxis.set_minor_locator(x_minorLocator)

# y_majorLocator = MultipleLocator(0.2)
# y_minorLocator = MultipleLocator(0.05)
# ax2.yaxis.set_major_locator(y_majorLocator)
# ax2.yaxis.set_minor_locator(y_minorLocator)

fig.tight_layout()
fig.savefig('./{0}_{1:.2f}_binary_lc.pdf'.format(target_star, target_binary_period))
plt.close(fig)
