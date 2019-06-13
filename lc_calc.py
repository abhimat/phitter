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

lambda_Kp = 2.124e-6 * u.m
dlambda_Kp = 0.351e-6 * u.m

lambda_H = 1.633e-6 * u.m
dlambda_H = 0.296e-6 * u.m

# Reference fluxes, calculated with PopStar
## Vega magnitudes (m_Vega = 0.03)
ks_filt_info = synthetic.get_filter_info('naco,Ks')
kp_filt_info = synthetic.get_filter_info('nirc2,Kp')
h_filt_info = synthetic.get_filter_info('nirc2,H')

flux_ref_Ks = ks_filt_info.flux0 * (u.erg / u.s) / (u.cm**2.)
flux_ref_Kp = kp_filt_info.flux0 * (u.erg / u.s) / (u.cm**2.)
flux_ref_H = h_filt_info.flux0 * (u.erg / u.s) / (u.cm**2.)

# Stellar Parameters
# stellar_params = (mass, rad, teff, mag_Kp, mag_H)

def single_star_lc(stellar_params, use_blackbody_atm=False,
                   num_triangles=1500):
    # Read in the stellar parameters of the current star
    (star_mass, star_rad, star_teff, star_mag_Kp, star_mag_H) = stellar_params
    
    err_out = np.array([-1.])

    # Set up a single star model
    sing_star = phoebe.default_star()
    
    # Light curve dataset
    if use_blackbody_atm:
        sing_star.add_dataset(phoebe.dataset.lc, times=[0], dataset='mod_lc_Kp', passband='Keck_NIRC2:Kp', ld_func='linear', ld_coeffs=[0.0])
        sing_star.add_dataset(phoebe.dataset.lc, times=[0], dataset='mod_lc_H', passband='Keck_NIRC2:H', ld_func='linear', ld_coeffs=[0.0])
    else:
        sing_star.add_dataset(phoebe.dataset.lc, times=[0], dataset='mod_lc_Kp', passband='Keck_NIRC2:Kp')
        sing_star.add_dataset(phoebe.dataset.lc, times=[0], dataset='mod_lc_H', passband='Keck_NIRC2:H')
    
    # Set up compute
    if use_blackbody_atm:
        sing_star.add_compute('phoebe', compute='detailed', distortion_method='sphere', irrad_method='none', atm='blackbody')
    else:
        sing_star.add_compute('phoebe', compute='detailed', distortion_method='sphere', irrad_method='none')
    
    
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
    sing_star_fluxes_Kp = np.array(sing_star['fluxes@lc@mod_lc_Kp@model'].value) * u.solLum / (4* np.pi * u.m**2)
    sing_star_mags_Kp = -2.5 * np.log10(sing_star_fluxes_Kp / flux_ref_Kp) + 0.03
    
    sing_star_fluxes_H = np.array(sing_star['fluxes@lc@mod_lc_H@model'].value) * u.solLum / (4* np.pi * u.m**2)
    sing_star_mags_H = -2.5 * np.log10(sing_star_fluxes_Kp / flux_ref_H) + 0.03
    
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
    (star1_mass, star1_rad, star1_teff, star1_mag_Kp, star1_mag_H) = star1_params
    (star2_mass, star2_rad, star2_teff, star2_mag_Kp, star2_mag_H) = star2_params
    
    # Read in the parameters of the binary system
    (binary_period, binary_ecc, binary_inc, t0) = binary_params
    
    err_out = (np.array([-1.]), np.array([-1.]))
    
    
    # Set up binary model
    b = phoebe.default_binary()
    
    ## Set period, semimajor axis, and mass ratio (q)
    binary_sma = ((binary_period**2. * const.G * (star1_mass + star2_mass)) / (4. * np.pi**2.))**(1./3.)
    
    binary_q = star2_mass / star1_mass
    
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
        print(np.abs((star1_rad - star1_rad_max) / star1_rad_max))
        print(np.abs((star2_rad - star2_rad_max) / star2_rad_max))
    
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
        print(star1_semidetached)
        print(star2_semidetached)
        print(star1_overflow)
        print(star2_overflow)
    
    ## If none of these overflow cases, set variable to store if binary is detached
    binary_detached = (not star1_semidetached) and (not star2_semidetached) and (not star1_overflow) and (not star2_overflow)
    
    ### Set non-zero eccentricity only if binary is detached
    if binary_detached:
        b.set_value('ecc@binary@component', binary_ecc)
    else:
        b.set_value('ecc@binary@component', 0.)
    
    ## Change set up for contact or semidetached cases
    if star1_overflow or star2_overflow:
        b = phoebe.default_binary(contact_binary=True)
        
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
    (kp_MJDs, h_MJDs) = observation_times
    
    ## Phase the observation times
    kp_phased_days = ((kp_MJDs - t0) % binary_period.to(u.d).value) / binary_period.to(u.d).value
    h_phased_days = ((h_MJDs - t0) % binary_period.to(u.d).value) / binary_period.to(u.d).value
    
    # Add light curve datasets
    ## Kp
    kp_phases_sorted_inds = np.argsort(kp_phased_days)
    
    kp_model_times = (kp_phased_days) * binary_period.to(u.d).value
    kp_model_times = kp_model_times[kp_phases_sorted_inds]
    
    if use_blackbody_atm:
        b.add_dataset(phoebe.dataset.lc, time=kp_model_times, dataset='mod_lc_Kp', passband='Keck_NIRC2:Kp', ld_func='linear', ld_coeffs=[0.0])
    else:
        b.add_dataset(phoebe.dataset.lc, time=kp_model_times, dataset='mod_lc_Kp', passband='Keck_NIRC2:Kp')
    
    ## H
    h_phases_sorted_inds = np.argsort(h_phased_days)
    
    h_model_times = (h_phased_days) * binary_period.to(u.d).value
    h_model_times = h_model_times[h_phases_sorted_inds]
    
    if use_blackbody_atm:
        b.add_dataset(phoebe.dataset.lc, times=h_model_times, dataset='mod_lc_H', passband='Keck_NIRC2:H', ld_func='linear', ld_coeffs=[0.0])
    else:
        b.add_dataset(phoebe.dataset.lc, times=h_model_times, dataset='mod_lc_H', passband='Keck_NIRC2:H')
    
    # Add mesh dataset if making mesh plot
    if make_mesh_plots:
        b.add_dataset('mesh', times=[binary_period/4.], dataset='mod_mesh')
    
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
    ## Kp
    model_fluxes_Kp = np.array(b['fluxes@lc@mod_lc_Kp@model'].value) * u.solLum / (4* np.pi * u.m**2)   # * u.W / (u.m**2.)
    model_mags_Kp = -2.5 * np.log10(model_fluxes_Kp / flux_ref_Kp) + 0.03
    
    ## H
    model_fluxes_H = np.array(b['fluxes@lc@mod_lc_H@model'].value) * u.solLum / (4* np.pi * u.m**2)   # * u.W / (u.m**2.)
    model_mags_H = -2.5 * np.log10(model_fluxes_H / flux_ref_H) + 0.03
    
    return (model_mags_Kp, model_mags_H)
    
def phased_obs(observation_times, binary_period, t0):
    # Phase the observation times
    ## Read in observation times
    (kp_MJDs, h_MJDs) = observation_times
    
    ## Phase the observation times
    kp_phased_days = ((kp_MJDs - t0) % binary_period.to(u.d).value) / binary_period.to(u.d).value
    h_phased_days = ((h_MJDs - t0) % binary_period.to(u.d).value) / binary_period.to(u.d).value
    
    ## Kp
    kp_phases_sorted_inds = np.argsort(kp_phased_days)
    
    kp_model_times = (kp_phased_days) * binary_period.to(u.d).value
    kp_model_times = kp_model_times[kp_phases_sorted_inds]
    
    ## H
    h_phases_sorted_inds = np.argsort(h_phased_days)
    
    h_model_times = (h_phased_days) * binary_period.to(u.d).value
    h_model_times = h_model_times[h_phases_sorted_inds]
    
    return ((kp_phased_days, kp_phases_sorted_inds, kp_model_times),
            (h_phased_days, h_phases_sorted_inds, h_model_times))


def dist_ext_mag_calc(input_mags, target_dist, Kp_ext, H_ext):
    (mags_Kp, mags_H) = input_mags
    
    # App mag at target distance (default PHOEBE dist = 1m)
    mags_Kp = mags_Kp + 5. * np.log10(target_dist / (1. * u.m))
    mags_H = mags_H + 5. * np.log10(target_dist / (1. * u.m))
    
    # Add extinction
    mags_Kp = mags_Kp + Kp_ext
    mags_H = mags_H + H_ext
    
    # Return mags at target distance and extinction
    return (mags_Kp, mags_H)
    

def flux_adj(mags_pri, mags_ref_pri, mags_sec, mags_ref_sec, mags_bin):
    """Perform flux adjustment for binary magnitudes
    * Uses calculated and reference single star magnitudes for binary components
    * Derives adjustment to binary magnitude based on the discrepancy
    * Applies correction to the calculated binary magnitudes
    
    Keyword arguments:
    mags_pri -- Model calculated magnitudes (Kp, H) of primary
    mags_ref_pri -- Reference magnitudes (Kp, H) of primary
    mags_sec -- Model calculated magnitudes (Kp, H) of secondary
    mags_ref_sec -- Reference magnitudes (Kp, H) of secondary
    mags_bin -- Model calculated magnitudes ([Kp], [H]) of the binary system
    """
    
    # Properly expand and extract the input magnitudes
    (mag_pri_Kp, mag_pri_H) = mags_pri
    (mag_ref_pri_Kp, mag_ref_pri_H) = mags_ref_pri
    
    (mag_sec_Kp, mag_sec_H) = mags_sec
    (mag_ref_sec_Kp, mag_ref_sec_H) = mags_ref_sec
    
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
                     isoc_Ks_ext, Kp_ext, H_ext, ext_alpha,
                     isoc_dist, bin_dist,
                     use_blackbody_atm=False,
                     make_mesh_plots=False, plot_name=None,
                     num_triangles=1500):
    
    # Extinction law (using Nogueras-Lara+ 2018)
    ext_alpha = 2.30
    
    # Calculate extinctions implied by isochrone extinction
    isoc_Kp_ext = isoc_Ks_ext * (lambda_Ks / lambda_Kp)**ext_alpha
    isoc_H_ext = isoc_Ks_ext * (lambda_Ks / lambda_H)**ext_alpha
    
    # Calculate extinction adjustments
    Kp_ext_adj = (Kp_ext - isoc_Kp_ext)
    H_ext_adj = (H_ext - isoc_H_ext)
    
    # Calculate distance modulus adjustments
    dist_mod_mag_adj = 5. * np.log10(bin_dist / (isoc_dist.to(u.pc)).value)
    
    # Extract stellar parameters from input
    (star1_mass, star1_rad, star1_teff, star1_mag_Kp, star1_mag_H) = star1_params_lcfit
    (star2_mass, star2_rad, star2_teff, star2_mag_Kp, star2_mag_H) = star2_params_lcfit
    
    # Run single star model for reference flux calculations
    (star1_sing_mag_Kp, star1_sing_mag_H) = single_star_lc(star1_params_lcfit,
                                                use_blackbody_atm=use_blackbody_atm,
                                                num_triangles=num_triangles)
    
    if (star1_sing_mag_Kp[0] == -1.) or (star1_sing_mag_H[0] == -1.):
        return -np.inf
    
    (star2_sing_mag_Kp, star2_sing_mag_H) = single_star_lc(star2_params_lcfit,
                                                use_blackbody_atm=use_blackbody_atm,
                                                num_triangles=num_triangles)
    
    if (star2_sing_mag_Kp[0] == -1.) or (star2_sing_mag_H[0] == -1.):
        return -np.inf
    
    ## Apply distance modulus and isoc. extinction to single star magnitudes
    (star1_sing_mag_Kp, star1_sing_mag_H) = dist_ext_mag_calc(
                                                (star1_sing_mag_Kp,
                                                star1_sing_mag_H),
                                                isoc_dist,
                                                isoc_Kp_ext, isoc_H_ext)
    
    (star2_sing_mag_Kp, star2_sing_mag_H) = dist_ext_mag_calc(
                                                (star2_sing_mag_Kp,
                                                star2_sing_mag_H),
                                                isoc_dist,
                                                isoc_Kp_ext, isoc_H_ext)
    
    
    # Run binary star model to get binary mags
    (binary_mags_Kp, binary_mags_H) = binary_star_lc(
                                          star1_params_lcfit,
                                          star2_params_lcfit,
                                          binary_params,
                                          observation_times,
                                          use_blackbody_atm=use_blackbody_atm,
                                          make_mesh_plots=make_mesh_plots,
                                          plot_name=plot_name,
                                          num_triangles=num_triangles)
    if (binary_mags_Kp[0] == -1.) or (binary_mags_H[0] == -1.):
        return -np.inf
    
    ## Apply distance modulus and isoc. extinction to binary magnitudes
    (binary_mags_Kp, binary_mags_H) = dist_ext_mag_calc(
                                          (binary_mags_Kp, binary_mags_H),
                                          isoc_dist,
                                          isoc_Kp_ext, isoc_H_ext)
    
    # Apply flux correction to the binary magnitudes
    (binary_mags_Kp, binary_mags_H) = flux_adj(
                                          (star1_sing_mag_Kp, star1_sing_mag_H),
                                          (star1_mag_Kp, star1_mag_H),
                                          (star2_sing_mag_Kp, star2_sing_mag_H),
                                          (star2_mag_Kp, star2_mag_H),
                                          (binary_mags_Kp, binary_mags_H))
    
    # Apply the extinction difference between model and the isochrone values
    binary_mags_Kp += Kp_ext_adj
    binary_mags_H += H_ext_adj
    
    # Apply the distance modulus for difference between isoc. distance and bin. distance
    # (Same for each filter)
    binary_mags_Kp += dist_mod_mag_adj
    binary_mags_H += dist_mod_mag_adj
    
    # Return final light curve
    return (binary_mags_Kp, binary_mags_H)