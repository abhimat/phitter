#!/usr/bin/env python

# Light curve calculation functions,
# for fitting with PHOEBE
# ---
# Abhimat Gautam

import phoebe
from phoebe import u
from phoebe import c as const

import numpy as np

from popstar import synthetic

import sys

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.ticker import MultipleLocator

# Filter properties
lambda_Ks = 2.18e-6 * u.m
dlambda_Ks = 0.35e-6 * u.m

lambda_B = 445.0e-9 * u.m
dlambda_B = 94.0e-9 * u.m

lambda_R = 658.0e-9 * u.m
dlambda_R = 138.0e-9 * u.m

# Reference fluxes, calculated with PopStar
## Vega magnitudes (m_Vega = 0.03)
ks_filt_info = synthetic.get_filter_info('naco,Ks')
b_filt_info = synthetic.get_filter_info('ubv,B')
r_filt_info = synthetic.get_filter_info('ubv,R')

v_filt_info = synthetic.get_filter_info('ubv,V')

flux_ref_Ks = ks_filt_info.flux0 * (u.erg / u.s) / (u.cm**2.)
flux_ref_B = b_filt_info.flux0 * (u.erg / u.s) / (u.cm**2.)
flux_ref_R = r_filt_info.flux0 * (u.erg / u.s) / (u.cm**2.)

flux_ref_V = v_filt_info.flux0 * (u.erg / u.s) / (u.cm**2.)

# Stellar Parameters
# stellar_params = (mass, rad, teff, mag_B, mag_R, pblum_B, pblum_R)

def single_star_lc(stellar_params,
        use_blackbody_atm=False,
        num_triangles=1500):
    # Read in the stellar parameters of the current star
    (star_mass, star_rad, star_teff,
     star_mag_Kp, star_mag_H, star_pblum_Kp, star_pblum_H) = stellar_params
    
    err_out = np.array([-1.])

    # Set up a single star model
    sing_star = phoebe.default_star()
    
    # Light curve dataset
    if use_blackbody_atm:
        sing_star.add_dataset(phoebe.dataset.lc, times=[0], dataset='mod_lc_Kp', passband='Keck_NIRC2:Kp', ld_func='linear', ld_coeffs=[0.0])
        sing_star.add_dataset(phoebe.dataset.lc, times=[0], dataset='mod_lc_H', passband='Keck_NIRC2:H', ld_func='linear', ld_coeffs=[0.0])
        # sing_star.add_dataset(phoebe.dataset.lc, times=[0], dataset='mod_lc_V', passband='Johnson:V', ld_func='linear', ld_coeffs=[0.0])
    else:
        sing_star.add_dataset(phoebe.dataset.lc, times=[0], dataset='mod_lc_Kp', passband='Keck_NIRC2:Kp')
        sing_star.add_dataset(phoebe.dataset.lc, times=[0], dataset='mod_lc_H', passband='Keck_NIRC2:H')
        # sing_star.add_dataset(phoebe.dataset.lc, times=[0], dataset='mod_lc_V', passband='Johnson:V')
    
    # Set up compute
    if use_blackbody_atm:
        sing_star.add_compute('phoebe', compute='detailed', distortion_method='sphere', irrad_method='none', atm='blackbody')
    else:
        sing_star.add_compute('phoebe', compute='detailed', distortion_method='sphere', irrad_method='none')
    
    # Set a default distance
    sing_star.set_value('distance', 10 * u.pc)
    
    # Set the passband luminosities
    sing_star.set_value('pblum@mod_lc_Kp', star_pblum_Kp)
    sing_star.set_value('pblum@mod_lc_H', star_pblum_H)
    
    # Set the stellar parameters
    sing_star.set_value('teff@component', star_teff)
    sing_star.set_value('requiv@component', star_rad)
    
    # Set the number of triangles in the mesh
    sing_star.set_value('ntriangles@detailed@compute', num_triangles)
    
    # Run compute
    try:
        sing_star.run_compute(compute='detailed', model='run')
    except: # Catch errors during computation (probably shouldn't happen during individual star computation)
        print("Error during primary ind. star compute: {0}".format(sys.exc_info()[0]))
        return (err_out, err_out)
    
    # Retrieve computed fluxes from phoebe
    sing_star_fluxes_Kp = np.array(sing_star['fluxes@lc@mod_lc_Kp@model'].value) * u.W / (u.m**2)
    sing_star_mags_Kp = -2.5 * np.log10(sing_star_fluxes_Kp / flux_ref_Kp) + 0.03
    
    sing_star_fluxes_H = np.array(sing_star['fluxes@lc@mod_lc_H@model'].value) * u.W / (u.m**2)
    sing_star_mags_H = -2.5 * np.log10(sing_star_fluxes_H / flux_ref_H) + 0.03
        
    return (sing_star_mags_Kp, sing_star_mags_H)

def binary_star_lc(star1_params, star2_params, binary_params, observation_times,
        use_blackbody_atm=False, make_mesh_plots=False, plot_name=None,
        print_diagnostics=False, par_compute=False, num_par_processes=8,
        num_triangles=1500):
    """Compute the light curve for a binary system
    
    Keyword arguments:
    star1_params -- Tuple of parameters for the primary star
    star2_params -- Tuple of parameters for the secondary star
    binary_params -- Tuple of parameters for the binary system configuration
    observation_times -- Tuple of observation times,
        with numpy array of MJDs in each band
        (kp_MJDs, h_MJDs) = observation_times
    use_blackbody_atm -- Use blackbody atmosphere
        instead of default Castelli & Kurucz (default False)
    make_mesh_plots -- Make a mesh plot of the binary system (default False)
    plot_name
    print_diagnostics
    par_compute
    num_par_processes
    """
    
    
    if par_compute:
        # TODO: Need to implement parallelization correctly
        phoebe.mpi_on(nprocs=num_par_processes)
    else:
        phoebe.mpi_off()
    
    # Read in the stellar parameters of the binary components
    (star1_mass, star1_rad, star1_teff,
     star1_mag_B, star1_mag_R,
     star1_pblum_B, star1_pblum_R) = star1_params
    (star2_mass, star2_rad, star2_teff,
     star2_mag_B, star2_mag_R,
     star2_pblum_B, star2_pblum_R) = star2_params
    
    # Read in the parameters of the binary system
    (binary_period, binary_ecc, binary_inc, t0) = binary_params
    
    err_out = (np.array([-1.]), np.array([-1.]))
    
    
    # Set up binary model
    b = phoebe.default_binary()
    
    ## Set a default distance
    b.set_value('distance', 10 * u.pc)
    
    ## Set period, semimajor axis, and mass ratio (q)
    binary_sma = ((binary_period**2. * const.G * (star1_mass + star2_mass)) / (4. * np.pi**2.))**(1./3.)
    
    binary_q = star2_mass / star1_mass
    
    if print_diagnostics:
        print('\nBinary orbit checks')
        print('Binary SMA: {0}'.format(binary_sma.to(u.AU)))
        print('Binary Mass Ratio (q): {0}'.format(binary_q))
    
    b.set_value('period@orbit', binary_period)
    b.set_value('sma@binary@component', binary_sma)
    b.set_value('q@binary@component', binary_q)
    
    ## Inclination
    b.set_value('incl@orbit', binary_inc)
    
    # Check for overflow
    ## Variables to help store the non-detached binary cases
    star1_semidetached = False
    star2_semidetached = False
    
    star1_overflow = False
    star2_overflow = False
    
    ## Get the max radii for both component stars
    star1_rad_max = b.get_value('requiv_max@primary@component') * u.solRad
    star2_rad_max = b.get_value('requiv_max@secondary@component') * u.solRad
    
    ## Check for semidetached cases
    if print_diagnostics:
        print('\nSemidetached checks')
        print('Star 1: {0}'.format(np.abs((star1_rad - star1_rad_max) / star1_rad_max)))
        print('Star 2: {0}'.format(np.abs((star2_rad - star2_rad_max) / star2_rad_max)))
    
    semidet_cut = 0.001   # (within 0.1% of max radii)
    semidet_cut = 0.015   # (within 1.5% of max radii)
    
    if np.abs((star1_rad - star1_rad_max) / star1_rad_max) < semidet_cut:
        star1_semidetached = True
    if np.abs((star2_rad - star2_rad_max) / star2_rad_max) < semidet_cut:
        star2_semidetached = True
    
    ## Check for overflow
    if (star1_rad > star1_rad_max) and not star1_semidetached:
        star1_overflow = True
    
    if (star2_rad > star2_rad_max) and not star2_semidetached:
        star2_overflow = True
    
    ### Check for if both stars are overflowing; which star overflows more?
    ### Choose that star to be overflowing more
    if star1_overflow and star2_overflow:
        if (star1_rad - star1_rad_max) >= (star2_rad - star2_rad_max):
            star2_overflow = False
        else:
            star1_overflow = False
    
    
    if print_diagnostics:
        print('\nOverflow Checks')
        print('Star 1 Semidetached: {0}'.format(star1_semidetached))
        print('Star 2 Semidetached: {0}'.format(star2_semidetached))
        print('Star 1 Overflow: {0}'.format(star1_overflow))
        print('Star 2 Overflow: {0}'.format(star2_overflow))
    
    ## If none of these overflow cases, set variable to store if binary is detached
    binary_detached = (not star1_semidetached) and \
                      (not star2_semidetached) and \
                      (not star1_overflow) and \
                      (not star2_overflow)
    
    ### Set non-zero eccentricity only if binary is detached
    if binary_detached:
        b.set_value('ecc@binary@component', binary_ecc)
    else:
        b.set_value('ecc@binary@component', 0.)
    
    ## Change set up for contact or semidetached cases
    if star1_overflow or star2_overflow:
        b = phoebe.default_binary(contact_binary=True)
        
        ### Reset all necessary binary properties for contact system
        b.set_value('distance', 10 * u.pc)
        
        b.set_value('period@orbit', binary_period)
        b.set_value('sma@binary@component', binary_sma)
        b.set_value('q@binary@component', binary_q)
        b.set_value('incl@orbit', binary_inc)
    
    if star1_semidetached and not star2_overflow:
        b.add_constraint('semidetached', 'primary')
    if star2_semidetached and not star1_overflow:
        b.add_constraint('semidetached', 'secondary')
    
    # Set up compute
    if use_blackbody_atm:
        b.add_compute('phoebe', compute='detailed', irrad_method='wilson', atm='blackbody')
    else:
        b.add_compute('phoebe', compute='detailed', irrad_method='wilson')
    
    # Set the parameters of the component stars of the system
    ## Primary
    b.set_value('teff@primary@component', star1_teff)
    if (not star1_semidetached) and (not star2_overflow):
        b.set_value('requiv@primary@component', star1_rad)
    
    ## Secondary
    b.set_value('teff@secondary@component', star2_teff)
    if (not star2_semidetached) and (not star1_overflow):
        b.set_value('requiv@secondary@component', star2_rad)
    
    
    # Set the number of triangles in the mesh
    b.set_value('ntriangles@primary@detailed@compute', num_triangles)
    b.set_value('ntriangles@secondary@detailed@compute', num_triangles)
    
    # Phase the observation times
    ## Read in observation times
    (b_MJDs, r_MJDs) = observation_times
    
    ## Phase the observation times
    b_phased_days = ((b_MJDs - t0) % binary_period.to(u.d).value) / binary_period.to(u.d).value
    r_phased_days = ((r_MJDs - t0) % binary_period.to(u.d).value) / binary_period.to(u.d).value
    
    # Add light curve datasets
    ## B
    b_phases_sorted_inds = np.argsort(b_phased_days)
    
    b_model_times = (b_phased_days) * binary_period.to(u.d).value
    b_model_times = b_model_times[b_phases_sorted_inds]
    
    if use_blackbody_atm:
        b.add_dataset(phoebe.dataset.lc, time=b_model_times, dataset='mod_lc_B', passband='Johnson:B', ld_func='linear', ld_coeffs=[0.0])
    else:
        b.add_dataset(phoebe.dataset.lc, time=b_model_times, dataset='mod_lc_B', passband='Johnson:B')
    
    ## R
    r_phases_sorted_inds = np.argsort(r_phased_days)
    
    r_model_times = (r_phased_days) * binary_period.to(u.d).value
    r_model_times = r_model_times[r_phases_sorted_inds]
    
    if use_blackbody_atm:
        b.add_dataset(phoebe.dataset.lc, times=r_model_times, dataset='mod_lc_R', passband='Johnson:R', ld_func='linear', ld_coeffs=[0.0])
    else:
        b.add_dataset(phoebe.dataset.lc, times=r_model_times, dataset='mod_lc_R', passband='Johnson:R')
    
    # Add mesh dataset if making mesh plot
    if make_mesh_plots:
        b.add_dataset('mesh', times=[binary_period/4.], dataset='mod_mesh')
    
    # Set the passband luminosities for the stars
    b.set_value_all('pblum_ref', 'self')
    
    b.set_value('pblum@primary@mod_lc_B', star1_pblum_B)
    b.set_value('pblum@primary@mod_lc_R', star1_pblum_R)
    
    b.set_value('pblum@secondary@mod_lc_B', star2_pblum_B)
    b.set_value('pblum@secondary@mod_lc_R', star2_pblum_R)
    
    # Run compute
    # b.run_compute(compute='detailed', model='run')
    try:
        b.run_compute(compute='detailed', model='run')
    except:
        if print_diagnostics:
            print("Error during primary binary compute: {0}".format(sys.exc_info()[0]))
        return err_out
    
    
    # Save out mesh plot
    if make_mesh_plots:
        ## Plot Nerdery
        plt.rc('font', family='serif')
        plt.rc('font', serif='Computer Modern Roman')
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r"\usepackage{gensymb}")

        plt.rc('xtick', direction = 'in')
        plt.rc('ytick', direction = 'in')
        plt.rc('xtick', top = True)
        plt.rc('ytick', right = True)
        
        suffix_str = ''
        if plot_name is not None:
            suffix_str = '_' + plot_name
        
        ## Mesh plot
        b['mod_mesh@model'].plot(save='./binary_mesh{0}.pdf'.format(suffix_str))
    
    
    # Get fluxes
    ## B
    model_fluxes_B = np.array(b['fluxes@lc@mod_lc_B@model'].value) * u.W / (u.m**2.)
    model_mags_B = -2.5 * np.log10(model_fluxes_B / flux_ref_B) + 0.03
    
    ## R
    model_fluxes_R = np.array(b['fluxes@lc@mod_lc_R@model'].value) * u.W / (u.m**2.)
    model_mags_R = -2.5 * np.log10(model_fluxes_R / flux_ref_R) + 0.03
    
    if print_diagnostics:
        print('\nFlux Checks')
        print('Fluxes, B: {0}'.format(model_fluxes_B))
        print('Mags, B: {0}'.format(model_mags_B))
        print('Fluxes, R: {0}'.format(model_fluxes_R))
        print('Mags, R: {0}'.format(model_mags_R))
    
    return (model_mags_B, model_mags_R)
    
def phased_obs(observation_times, binary_period, t0):
    # Phase the observation times
    ## Read in observation times
    (b_MJDs, r_MJDs) = observation_times
    
    ## Phase the observation times
    b_phased_days = ((b_MJDs - t0) % binary_period.to(u.d).value) / binary_period.to(u.d).value
    r_phased_days = ((r_MJDs - t0) % binary_period.to(u.d).value) / binary_period.to(u.d).value
    
    ## B
    b_phases_sorted_inds = np.argsort(b_phased_days)
    
    b_model_times = (b_phased_days) * binary_period.to(u.d).value
    b_model_times = b_model_times[b_phases_sorted_inds]
    
    ## R
    r_phases_sorted_inds = np.argsort(r_phased_days)
    
    r_model_times = (r_phased_days) * binary_period.to(u.d).value
    r_model_times = r_model_times[r_phases_sorted_inds]
    
    return ((b_phased_days, b_phases_sorted_inds, b_model_times),
            (r_phased_days, r_phases_sorted_inds, r_model_times))


def dist_ext_mag_calc(input_mags, target_dist, B_ext, R_ext):
    (mags_B, mags_R) = input_mags
    
    # App mag at target distance (default system dist = 10 pc)
    mags_B = mags_B + 5. * np.log10(target_dist / (10. * u.pc))
    mags_R = mags_R + 5. * np.log10(target_dist / (10. * u.pc))
    
    # Add extinction
    mags_B = mags_B + B_ext
    mags_R = mags_R + R_ext
    
    # Return mags at target distance and extinction
    return (mags_B, mags_R)
    

def flux_adj(mags_pri, mags_ref_pri, mags_sec, mags_ref_sec, mags_bin):
    """Perform flux adjustment for binary magnitudes
    * Uses calculated and reference single star magnitudes for binary components
    * Derives adjustment to binary magnitude based on the discrepancy
    * Applies correction to the calculated binary magnitudes
    
    Keyword arguments:
    mags_pri -- Model calculated magnitudes (B, R) of primary
    mags_ref_pri -- Reference magnitudes (B, R) of primary
    mags_sec -- Model calculated magnitudes (B, R) of secondary
    mags_ref_sec -- Reference magnitudes (B, R) of secondary
    mags_bin -- Model calculated magnitudes ([B], [R]) of the binary system
    """
    
    # Properly expand and extract the input magnitudes
    (mag_pri_B, mag_pri_R) = mags_pri
    (mag_ref_pri_B, mag_ref_pri_R) = mags_ref_pri
    
    (mag_sec_B, mag_sec_R) = mags_sec
    (mag_ref_sec_B, mag_ref_sec_R) = mags_ref_sec
    
    (mags_bin_Kp, mags_bin_H) = mags_bin
    
    # Calculate flux adjustment for each component star
    flux_adj_star_pri_Kp = (10.**((mag_pri_Kp[0] - mag_ref_pri_Kp)/2.5))
    flux_adj_star_pri_H = (10.**((mag_pri_H[0] - mag_ref_pri_H)/2.5))
    
    flux_adj_star_sec_Kp = (10.**((mag_sec_Kp[0] - mag_ref_sec_Kp)/2.5))
    flux_adj_star_sec_H = (10.**((mag_sec_H[0] - mag_ref_sec_H)/2.5))
    
    # Calculate total flux adjustment for the binary system
    flux_adj_bin_Kp = ((10.**(mag_ref_pri_Kp / -2.5)) + (10.**(mag_ref_sec_Kp / -2.5))) / ((10.**(mag_pri_Kp[0] / -2.5)) + (10.**(mag_sec_Kp[0] / -2.5)))
    flux_adj_bin_H = ((10.**(mag_ref_pri_H / -2.5)) + (10.**(mag_ref_sec_H / -2.5))) / ((10.**(mag_pri_H[0] / -2.5)) + (10.**(mag_sec_H[0] / -2.5)))
    
    
    # Apply flux adjustment to the input binary magnitudes
    ## Convert magnitudes back into flux space
    fluxes_bin_Kp = flux_ref_Kp * (10.**((mags_bin_Kp - 0.03)/ -2.5))
    fluxes_bin_H = flux_ref_H * (10.**((mags_bin_H - 0.03)/ -2.5))
    
    ## Apply flux adjustment, derived from the single stars
    fluxes_bin_Kp = fluxes_bin_Kp * flux_adj_bin_Kp
    fluxes_bin_H = fluxes_bin_H * flux_adj_bin_H
    
    ## Convert to magnitudes and return
    adj_mags_bin_Kp = -2.5 * np.log10(fluxes_bin_Kp / flux_ref_Kp) + 0.03
    adj_mags_bin_H = -2.5 * np.log10(fluxes_bin_H / flux_ref_H) + 0.03
    
    return (adj_mags_bin_Kp, adj_mags_bin_H)


def binary_mags_calc(star1_params_lcfit, star2_params_lcfit,
                     binary_params,
                     observation_times,
                     isoc_Ks_ext, B_ext, R_ext, ext_alpha,
                     isoc_dist, bin_dist,
                     use_blackbody_atm=False,
                     make_mesh_plots=False, plot_name=None,
                     num_triangles=1500,
                     print_diagnostics=False):
    
    # Extinction law (using Nogueras-Lara+ 2018)
    ext_alpha = 2.30
    
    # Calculate extinctions implied by isochrone extinction
    isoc_B_ext = isoc_Ks_ext * (lambda_Ks / lambda_B)**ext_alpha
    isoc_R_ext = isoc_Ks_ext * (lambda_Ks / lambda_R)**ext_alpha
    
    # Calculate extinction adjustments
    B_ext_adj = (B_ext - isoc_B_ext)
    R_ext_adj = (R_ext - isoc_R_ext)
    
    # Calculate distance modulus adjustments
    dist_mod_mag_adj = 5. * np.log10(bin_dist / (isoc_dist.to(u.pc)).value)
    
    # Extract stellar parameters from input
    (star1_mass, star1_rad, star1_teff,
     star1_mag_B, star1_mag_R, star1_pblum_B, star1_pblum_R) = star1_params_lcfit
    (star2_mass, star2_rad, star2_teff,
     star2_mag_B, star2_mag_R, star2_pblum_B, star2_pblum_R) = star2_params_lcfit
    
    
    # Run binary star model to get binary mags
    (binary_mags_B, binary_mags_R) = binary_star_lc(
                                         star1_params_lcfit,
                                         star2_params_lcfit,
                                         binary_params,
                                         observation_times,
                                         use_blackbody_atm=use_blackbody_atm,
                                         make_mesh_plots=make_mesh_plots,
                                         plot_name=plot_name,
                                         num_triangles=num_triangles,
                                         print_diagnostics=print_diagnostics)
    if (binary_mags_B[0] == -1.) or (binary_mags_R[0] == -1.):
        return -np.inf
    
    ## Apply distance modulus and isoc. extinction to binary magnitudes
    (binary_mags_B, binary_mags_R) = dist_ext_mag_calc(
                                         (binary_mags_B, binary_mags_R),
                                         isoc_dist,
                                         isoc_B_ext, isoc_R_ext)
    
    # Apply the extinction difference between model and the isochrone values
    binary_mags_B += B_ext_adj
    binary_mags_R += R_ext_adj
    
    # Apply the distance modulus for difference between isoc. distance and bin. distance
    # (Same for each filter)
    binary_mags_B += dist_mod_mag_adj
    binary_mags_R += dist_mod_mag_adj
    
    # Return final light curve
    return (binary_mags_B, binary_mags_R)