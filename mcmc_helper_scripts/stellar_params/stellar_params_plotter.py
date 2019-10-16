#!/usr/bin/env python

# Stellar Params Plotter
# ---
# Abhimat Gautam

import numpy as np

from astropy import units as u
from astropy import constants as const
from astropy.table import Table

import scipy.stats as stats

import cPickle as pickle

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.ticker import MultipleLocator

# Read in data
trial_num = 1
num_walkers = 500
burn_ignore_len = 500
rows_ignore = num_walkers*burn_ignore_len


# Read in table of parameters
params_table = Table.read('./stellar_params.hdf5', path='data')
num_samples = len(params_table)

## Read in stellar parameters
K_ext_samps = params_table['K_ext'][rows_ignore:]
star1_rad_samps = params_table['star1_rad'][rows_ignore:]
star2_rad_samps = params_table['star2_rad'][rows_ignore:]
binary_inc_samps = params_table['binary_inc'][rows_ignore:]
binary_per_samps = params_table['binary_per'][rows_ignore:]
binary_dist_samps = params_table['binary_dist'][rows_ignore:]
t0_samps = params_table['t0'][rows_ignore:]

# Other stellar and binary parameters to derive
## Stellar parameters, to be derived from isochrone
star1_mass_init_samps = params_table['star1_mass_init'][rows_ignore:]
star1_mass_samps = params_table['star1_mass_init'][rows_ignore:]
star1_lum_samps = params_table['star1_lum'][rows_ignore:]
star1_teff_samps = params_table['star1_teff'][rows_ignore:]
star1_mag_Kp_samps = params_table['star1_mag_Kp'][rows_ignore:]
star1_mag_H_samps = params_table['star1_mag_H'][rows_ignore:]
star1_pblum_Kp_samps = params_table['star1_pblum_Kp'][rows_ignore:]
star1_pblum_H_samps = params_table['star1_pblum_H'][rows_ignore:]

star2_mass_init_samps = params_table['star2_mass_init'][rows_ignore:]
star2_mass_samps = params_table['star2_mass'][rows_ignore:]
star2_lum_samps = params_table['star2_lum'][rows_ignore:]
star2_teff_samps = params_table['star2_teff'][rows_ignore:]
star2_mag_Kp_samps = params_table['star2_mag_Kp'][rows_ignore:]
star2_mag_H_samps = params_table['star2_mag_H'][rows_ignore:]
star2_pblum_Kp_samps = params_table['star2_pblum_Kp'][rows_ignore:]
star2_pblum_H_samps = params_table['star2_pblum_H'][rows_ignore:]

## Binary parameters
binary_sma_samps = params_table['binary_sma'][rows_ignore:]
binary_sma_samps_solRad = params_table['binary_sma'][rows_ignore:].to(u.solRad).value
binary_q_samps = params_table['binary_q'][rows_ignore:]
binary_q_init_samps = params_table['binary_q_init'][rows_ignore:]

## Create inclination samples that are just between 0 and 90 degrees
binary_inc_90max_samps = binary_inc_samps.copy()

gt90_filt = np.where(binary_inc_samps > 90.)
binary_inc_90max_samps[gt90_filt] = 180.*u.deg - binary_inc_90max_samps[gt90_filt]


# Make plots for each quantity
## Function to find peaks in KDE
def param_peak_writer(param_name, param_vals, param_kde):
    ### Find peaks
    peak_param_vals = []
    peak_param_kdes = []
    
    for cur_index in range(len(param_vals)):
        #### Check that this is local max
        if cur_index - 1 >= 0:
            if param_kde[cur_index - 1] > param_kde[cur_index]:
                continue

        if cur_index + 1 < len(param_vals):
            if param_kde[cur_index + 1] > param_kde[cur_index]:
                continue
        
        #### If local max tests passed, append to lists
        peak_param_vals.append(param_vals[cur_index])
        peak_param_kdes.append(param_kde[cur_index])
    
    ### Write out peaks
    with open('./{0}_peaks_try{1}.txt'.format(param_name, trial_num), 'w') as peaks_file:
        peaks_file.write('{0:>10} {1:>10}\n'.format('Peak Sample', 'Peak KDE'))
        for cur_peak_val, cur_peak_kde in zip(peak_param_vals, peak_param_kdes):
            peaks_file.write('{0:>10f} {1:>10f}\n'.format(cur_peak_val, cur_peak_kde))
    
    return (peak_param_vals, peak_param_kdes)
    

def param_samps_plotter(samples, param_name, param_name_plot, param_unit_plot, plot_kde=True):
    ## Plot Nerdery
    plt.rc('font', family='serif')
    plt.rc('font', serif='Computer Modern Roman')
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r"\usepackage{gensymb}")
    
    plt.rc('xtick', direction = 'in')
    plt.rc('ytick', direction = 'in')
    plt.rc('xtick', top = True)
    plt.rc('ytick', right = True)
    
    if param_unit_plot != '':
        param_unit_plot = r" " + param_unit_plot
    
    ## Calculate KDE
    kde_calc = stats.gaussian_kde(samples)
    param_range_vals = np.linspace(np.min(samples), np.max(samples), 500)
    param_range_kde = kde_calc(param_range_vals)
    
    (peak_param_vals, peak_param_kdes) = param_peak_writer(param_name, param_range_vals, param_range_kde)
    
    ## Calculate percentiles for sample
    param_percentiles = np.percentile(samples, [15.866, 50., 84.134])
    param_med = param_percentiles[1]
    param_hi_unc = param_percentiles[2] - param_med
    param_lo_unc = param_med - param_percentiles[0]
    
    ## Set up figure
    fig = plt.figure(figsize=(4,2))
    
    ax = fig.add_subplot(1,1,1)
    
    ## Plot median and uncertainties
    ax.axvline(x=param_med, ls='-', color='k', alpha=0.8)
    ax.axvline(x=param_med + param_hi_unc, ls='--', color='k', alpha=0.8)
    ax.axvline(x=param_med - param_lo_unc, ls='--', color='k', alpha=0.8)
    
    ## Plot samples
    hist_n, hist_bins, hist_patches = ax.hist(samples, color='k', histtype='step', bins=20, density=True)
    ax.set_xlabel(r"{0}{1}".format(param_name_plot, param_unit_plot))
    
    if plot_kde:
        ax.plot(param_range_vals, param_range_kde, '-', color='royalblue', alpha=0.8, lw=1.5)
        ax.plot(peak_param_vals, peak_param_kdes, 'o', color='royalblue', alpha=0.8)
    
    ylims = ax.get_ylim()
    
    ## Parameter median and uncertainties text
    param_text = param_name_plot + r"$= " + '{0:.3f}'.format(param_med)
    param_text = param_text + r"^{+" + '{0:.3f}'.format(param_hi_unc)
    param_text = param_text + r"}_{-" + '{0:.3f}'.format(param_lo_unc) + r"}$"
    param_text = param_text + param_unit_plot
    
    if hist_bins[-1] - param_med > param_med - hist_bins[0]:
        ax.text(hist_bins[-1], 0.9 * (ylims[1] - ylims[0]) + ylims[0],
            param_text, ha='right', va='top', size='x-small')
    else:
        ax.text(hist_bins[0], 0.9 * (ylims[1] - ylims[0]) + ylims[0],
            param_text, ha='left', va='top', size='x-small')
    
    ## Save out plot
    fig.tight_layout()
    fig.savefig('./{0}_samples_try{1}.pdf'.format(param_name, trial_num))
    plt.close(fig)


param_samps_plotter(star1_rad_samps, 'star1_rad', r"$R_{1}$", r"$(R_{\odot})$", plot_kde=False)
param_samps_plotter(star1_mass_samps, 'star1_mass', r"$M_{1}$", r"$(M_{\odot})$", plot_kde=False)
param_samps_plotter(star1_mass_init_samps, 'star1_mass_init', r"Init. $M_{1}$", r"$(M_{\odot})$", plot_kde=False)
param_samps_plotter(star1_lum_samps, 'star1_lum', r"$L_{1}$", r"$(L_{\odot})$", plot_kde=False)
param_samps_plotter(star1_teff_samps, 'star1_teff', r"Eff. $T_{1}$", r"(K)", plot_kde=False)
param_samps_plotter(star1_mag_Kp_samps, 'star1_mag_Kp', r"$m_{K'}$", '', plot_kde=False)
param_samps_plotter(star1_mag_H_samps, 'star1_mag_H', r"$m_{H}$", '', plot_kde=False)

param_samps_plotter(star2_rad_samps, 'star2_rad', r"$R_{2}$", r"$(R_{\odot})$", plot_kde=False)
param_samps_plotter(star2_mass_samps, 'star2_mass', r"$M_{2}$", r"$(M_{\odot})$", plot_kde=False)
param_samps_plotter(star2_mass_init_samps, 'star2_mass_init', r"Init. $M_{2}$", r"$(M_{\odot})$", plot_kde=False)
param_samps_plotter(star2_lum_samps, 'star2_lum', r"$L_{2}$", r"$(L_{\odot})$", plot_kde=False)
param_samps_plotter(star2_teff_samps, 'star2_teff', r"Eff. $T_{2}$", r"(K)", plot_kde=False)
param_samps_plotter(star2_mag_Kp_samps, 'star2_mag_Kp', r"$m_{K'}$", '', plot_kde=False)
param_samps_plotter(star2_mag_H_samps, 'star2_mag_H', r"$m_{H}$", '', plot_kde=False)

param_samps_plotter(binary_sma_samps, 'binary_sma', r"$a$", r"(AU)", plot_kde=False)
param_samps_plotter(binary_sma_samps_solRad, 'binary_sma_solRad', r"$a$", r"$(R_{\odot})$", plot_kde=False)
param_samps_plotter(binary_inc_samps, 'binary_inc', r"$i$", r"$(\degree)$", plot_kde=False)
param_samps_plotter(binary_inc_90max_samps, 'binary_inc_90max', r"$i$", r"$(\degree)$", plot_kde=False)
param_samps_plotter(binary_q_samps, 'binary_q', r"$q = M_{2} / M_{1}$", '', plot_kde=False)
param_samps_plotter(binary_q_init_samps, 'binary_q_init', r"Init. $M_{2} / M_{1}$", '', plot_kde=False)
param_samps_plotter(binary_dist_samps/1000., 'binary_dist', r"$d$", r"(kpc)", plot_kde=False)


param_samps_plotter(K_ext_samps, 'K_ext', r"$A_{K'}$", '', plot_kde=False)

param_samps_plotter(binary_per_samps, 'period', "Period", '(d)', plot_kde=False)
param_samps_plotter(t0_samps, 't0', r"$t_0$", '(MJD)', plot_kde=False)



