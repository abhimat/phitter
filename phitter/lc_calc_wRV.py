#!/usr/bin/env python

# Light curve calculation functions,
# for fitting with PHOEBE
# ---
# Abhimat Gautam

import phoebe
from phoebe import u
from phoebe import c as const
from spisea import synthetic
from . import filters
import numpy as np
import sys
import copy
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.ticker import MultipleLocator

# Filters for default filter list
kp_filt = filters.nirc2_kp_filt()
h_filt = filters.nirc2_h_filt()

# Stellar Parameters
# stellar_params = (mass, rad, teff, mag_Kp, mag_H, pblum_Kp, pblum_H)

def single_star_lc(stellar_params,
        use_blackbody_atm=False,
        num_triangles=1500):
    # Read in the stellar parameters of the current star
    (star_mass, star_rad, star_teff, star_logg,
     [star_mag_Kp, star_mag_H],
     [star_pblum_Kp, star_pblum_H]) = stellar_params
    
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
        sing_star.run_compute(compute='detailed', model='run',
                              progressbar=False)
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
        filts_list=[kp_filt, h_filt],
        use_blackbody_atm=False,
        use_compact_object=False,
        irrad_frac_refl=1.0,
        make_mesh_plots=False, mesh_temp=False, mesh_temp_cmap=None,
        plot_name=None,
        print_diagnostics=False, par_compute=False, num_par_processes=8,
        num_triangles=1500):
    """Compute the light curve for a binary system
    
    Parameters
    ----------
    star1_params : tuple
        Tuple of parameters for the primary star
    star2_params : tuple
        Tuple of parameters for the secondary star
    binary_params : tuple
        Tuple of parameters for the binary system configuration
    observation_times : tuple of numpy arrays
        Tuple of observation times, with tuple length equal to [number of
        photometric filters] + [1: for RV observation times].
        Expects an iterable list or 1d numpy array of MJDs for each band
        and for the RVs.
        For example for photometry in Kp and H:
        (kp_MJDs, h_MJDs, rv_MJDs) = observation_times
    use_blackbody_atm : bool, default=False
        Use blackbody atmosphere instead of default Castelli & Kurucz
        atmosphere. Default: False (i.e.: using a C&K atm by default)
    use_compact_object : bool, default=False
        If true, sets eclipse_method to 'only_horizon', necessary for compact
        companions without eclipses. Default: False
    make_mesh_plots : bool, default=False
        Make a mesh plot of the binary system (default False)
    plot_name : str, default=None
    print_diagnostics : bool, default=False
    par_compute : bool, default=False
    num_par_processes : int, default=8
    """
    
    if par_compute:
        # TODO: Need to implement parallelization correctly
        phoebe.mpi_on(nprocs=num_par_processes)
    else:
        phoebe.mpi_off()
    
    # Read in the stellar parameters of the binary components
    (star1_mass, star1_rad, star1_teff, star1_logg,
     star1_filt_mags, star1_filt_pblums) = star1_params
    (star2_mass, star2_rad, star2_teff, star2_logg,
     star2_filt_mags, star2_filt_pblums) = star2_params
    
    # Read in the parameters of the binary system
    (binary_period, binary_ecc, binary_inc, t0) = binary_params
    
    err_out = (np.array([-1.]), np.array([-1.]),
               np.array([-1.]), np.array([-1.]))
    if make_mesh_plots:
        err_out = (np.array([-1.]), np.array([-1.]),
                   np.array([-1.]), np.array([-1.]),
                   np.array([-1.]))
    
    # Check for high temp ck2004 atmosphere limits
    if not use_blackbody_atm:
        # log g = 3.5, high temp bounds (above 31e3 K)
        if star1_teff > (31000 * u.K) and (4.0 > star1_logg > 3.5):
            star1_teff_round = 30995.0 * u.K
            
            if print_diagnostics:
                print('Star 1 out of C&K 2004 grid')
                print('star1_logg = {0:.4f}'.format(star1_logg))
                print('Rounding down star1_teff')
                print('{0:.4f} -> {1:.4f}'.format(star1_teff, star1_teff_round))
            
            star1_teff = star1_teff_round
        if star2_teff > (31000 * u.K) and (4.0 > star2_logg > 3.5):
            star2_teff_round = 30995.0 * u.K
            
            if print_diagnostics:
                print('Star 2 out of C&K 2004 grid')
                print('star2_logg = {0:.4f}'.format(star2_logg))
                print('Rounding down star2_teff')
                print('{0:.4f} -> {1:.4f}'.format(star2_teff, star2_teff_round))
            
            star2_teff = star2_teff_round
        
        # log g = 4.0, high temp bounds (above 40e3 K)
        if star1_teff > (40000 * u.K) and (4.5 > star1_logg > 4.0):
            star1_teff_round = 39995.0 * u.K
            
            print('Star 1 out of C&K 2004 grid')
            print('star1_logg = {0:.4f}'.format(star1_logg))
            print('Rounding down star1_teff')
            print('{0:.4f} -> {1:.4f}'.format(star1_teff, star1_teff_round))
            
            star1_teff = star1_teff_round
        if star2_teff > (40000 * u.K) and (4.0 > star2_logg > 3.50):
            star2_teff_round = 39995.0 * u.K
            
            print('Star 2 out of C&K 2004 grid')
            print('star2_logg = {0:.4f}'.format(star2_logg))
            print('Rounding down star2_teff')
            print('{0:.4f} -> {1:.4f}'.format(star2_teff, star2_teff_round))
            
            star2_teff = star2_teff_round
    
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
    
    # Star rotation support tests
    # print(b.get_value('period@secondary@star@component'))
    # print(b.get_value('period@secondary@star@constraint'))
    # b.remove_constraint('period@secondary@star@constraint')
    # b.set_value('period@secondary@star@component', 3 * u.d)
    # print(b.get_value('period@secondary'))
    # print(b.get_value('period@orbit'))
    # # print(hi)
    
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
    
    # If star 2 is overflowing, have to re set up model:
    # Calling this same binary_star_lc function again,
    # with star 2 as primary and star 1 as secondary.
    # Change t0 = t0 - per/2 to make sure phase is correct,
    # wrt stars 1 and 2 being in same respective position
    if star2_overflow and not star1_overflow:
        redo_binary_params = (binary_period, binary_ecc, binary_inc,
                              t0 - (binary_period.to(u.d).value/2.))
        
        return binary_star_lc(star2_params, star1_params,
                    redo_binary_params,
                    observation_times,
                    filts_list=filts_list,
                    use_blackbody_atm=use_blackbody_atm,
                    make_mesh_plots=make_mesh_plots,
                    mesh_temp=mesh_temp, mesh_temp_cmap=mesh_temp_cmap,
                    plot_name=plot_name,
                    print_diagnostics=print_diagnostics,
                    par_compute=par_compute, num_par_processes=num_par_processes,
                    num_triangles=num_triangles)
    
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
        b.add_compute('phoebe', compute='detailed',
                      irrad_method='wilson', atm='blackbody')
        
        # b.add_compute('phoebe', compute='detailed',
        #               irrad_method='horvat', atm='blackbody', distortion_method='rotstar')
        
        b.set_value('atm@primary@detailed', 'blackbody')
        b.set_value('atm@secondary@detailed', 'blackbody')
    else:
        b.add_compute('phoebe', compute='detailed', irrad_method='wilson')
    
    # Set the parameters of the component stars of the system
    ## Primary
    b.set_value('teff@primary@component', star1_teff)
    # b.set_value('logg@primary@component', star1_logg)
    if (not star1_semidetached) and (not star2_overflow):
        b.set_value('requiv@primary@component', star1_rad)
    
    ## Secondary
    b.set_value('teff@secondary@component', star2_teff)
    # b.set_value('logg@secondary@component', star2_logg)
    if (not star2_semidetached) and (not star1_overflow) and (not star2_overflow):
        try:
            b.set_value('requiv@secondary@component', star2_rad)
        except:
            print('\nOverflow Checks')
            print('Star 1 Semidetached: {0}'.format(star1_semidetached))
            print('Star 2 Semidetached: {0}'.format(star2_semidetached))
            print('Star 1 Overflow: {0}'.format(star1_overflow))
            print('Star 2 Overflow: {0}'.format(star2_overflow))
            
            print("Cannot set secondary radius: {0}".format(sys.exc_info()[0]))
            
            return err_out
    
    if print_diagnostics:
       print('Contact envelope parameters:')
       print(b.filter('contact_envelope@detailed@compute'))
    
    # Set the number of triangles in the mesh
    # Detached stars
    if len(b.filter('ntriangles@primary@detailed@compute')) == 1: 
        b.set_value('ntriangles@primary@detailed@compute', num_triangles)
        
    if len(b.filter('ntriangles@secondary@detailed@compute')) == 1: 
        b.set_value('ntriangles@secondary@detailed@compute', num_triangles)
    
    # Contact envelope
    if len(b.filter('ntriangles@contact_envelope@detailed@compute')) == 1:
       if print_diagnostics:
           print('Setting number of triangles for contact envelope')
       b.set_value('ntriangles@contact_envelope@detailed@compute',
                   num_triangles * 2.)
    
    # Phase the observation times
    num_filts = len(filts_list)
    
    # Read in observation times, and separate photometry from RVs
    filt_MJDs = observation_times[:num_filts]
    rv_MJDs = observation_times[num_filts]
    
    # Phase the observation times
    
    # Phase photometric observation times
    filt_phased_days = ()
    for filt_index in range(num_filts):
        # Phase the current filter's MJDs
        cur_filt_MJDs = filt_MJDs[filt_index]
        cur_filt_phased_days = (((cur_filt_MJDs - t0) %
                                 binary_period.to(u.d).value) /
                                binary_period.to(u.d).value)
        
        # Append to tuple of all filters' phased days
        filt_phased_days = filt_phased_days + (cur_filt_phased_days, )
    
    # Phase RV observation times
    rv_phased_days = (((rv_MJDs - t0) % binary_period.to(u.d).value) /
                      binary_period.to(u.d).value)
    
    
    # Add light curve datasets
    
    # Parameters to change when using a blackbody atmosphere
    if use_blackbody_atm:
        b.set_value_all('ld_mode_bol', 'manual')
        b.set_value_all('ld_func_bol', 'linear')
        b.set_value_all('ld_coeffs_bol', [0.0])
    
    # Set irradiation reflection fraction 
    b.set_value_all('irrad_frac_refl_bol', irrad_frac_refl)
    
    # Check for compact companion
    if use_compact_object:
        b.set_value('irrad_method@detailed', 'none')
    
    
    # Go through each filter and add a lightcurve dataset
    for (filt_index, filt) in enumerate(filts_list):
        filt_phases_sorted_inds = np.argsort(filt_phased_days[filt_index])
        
        filt_model_times = (filt_phased_days[filt_index] *
                            binary_period.to(u.d).value)
        filt_model_times = filt_model_times[filt_phases_sorted_inds]
        
        
        b.add_dataset(phoebe.dataset.lc, time=filt_model_times,
                      dataset=filt.phoebe_ds_name,
                      passband=filt.phoebe_pb_name)
        if use_blackbody_atm:
            b.set_value_all('ld_mode@' + filt.phoebe_ds_name, 'manual')
    
            b.set_value_all('ld_func@' + filt.phoebe_ds_name, 'linear')
    
            b.set_value_all('ld_coeffs@' + filt.phoebe_ds_name, [0.0])
    
    # Add RV dataset
    rv_phases_sorted_inds = np.argsort(rv_phased_days)
    
    rv_model_times = (rv_phased_days) * binary_period.to(u.d).value
    rv_model_times = rv_model_times[rv_phases_sorted_inds]
    
    # Uses passband of first filter in filts_list for calculating RVs
    b.add_dataset(phoebe.dataset.rv, time=rv_model_times,
                  dataset='mod_rv',
                  passband=filts_list[0].phoebe_pb_name)
    if use_blackbody_atm:
        b.set_value_all('ld_mode@mod_rv', 'manual')
        
        b.set_value_all('ld_func@mod_rv', 'linear')
        
        b.set_value_all('ld_coeffs@mod_rv', [0.0])    
    
    # Add mesh dataset if making mesh plot
    if make_mesh_plots:
        b.add_dataset('mesh', times=[(binary_period/4.).to(u.d).value],
                      dataset='mod_mesh')
        b.set_value('coordinates@mesh', ['uvw'])
        if mesh_temp:
            mesh_columns = ['us', 'vs', 'ws', 'rprojs',
                            'teffs', 'loggs', 'rs', 'areas']
            for filt in filts_list:
                mesh_columns.append('*@' + filt.phoebe_ds_name)
            
            b['columns@mesh'] = mesh_columns
    
    # Set the passband luminosities for the stars
    if (not star1_overflow) and (not star2_overflow):
        # Detached / semidetached case
        for (filt_index, filt) in enumerate(filts_list):
            b.set_value('pblum_mode@' + filt.phoebe_ds_name,
                        'decoupled')
            
            b.set_value('pblum@primary@' + filt.phoebe_ds_name,
                        star1_filt_pblums[filt_index])
            
            b.set_value('pblum@secondary@' + filt.phoebe_ds_name,
                        star2_filt_pblums[filt_index])
    else:
        if star1_overflow:
            for (filt_index, filt) in enumerate(filts_list):
                b.set_value('pblum@primary@' + filt.phoebe_ds_name,
                            star1_filt_pblums[filt_index])
        elif star2_overflow:
            for (filt_index, filt) in enumerate(filts_list):
                b.set_value('pblum@secondary@' + filt.phoebe_ds_name,
                            star2_filt_pblums[filt_index])
    
    # Run compute
    # Determine eclipse method
    if use_compact_object:
        eclipse_method = 'only_horizon'
    else:
        eclipse_method = 'native'
    
    if print_diagnostics:
        print("Trying inital compute run")
        b.run_compute(compute='detailed', model='run',
                      progressbar=False, eclipse_method=eclipse_method)
    else:
        try:
            b.run_compute(compute='detailed', model='run',
                          progressbar=False, eclipse_method=eclipse_method)
        except:
            if print_diagnostics:
                print("Error during primary binary compute: {0}".format(
                            sys.exc_info()[0])
                     )
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
        # plt.rc('xtick', top = True)
        # plt.rc('ytick', right = True)
        
        suffix_str = ''
        if plot_name is not None:
            suffix_str = '_' + plot_name
        
        ## Mesh plot
        if mesh_temp:
            mesh_plot_out = b['mod_mesh@model'].plot(
                                save='./binary_mesh{0}.pdf'.format(suffix_str),
                                fc='teffs',
                                fcmap=mesh_temp_cmap,
                                ec='none',
                            )
                                                     
            # print(mesh_plot_out.axs)
            
            # Extract and output mesh quantities
            mesh_quant_names = ['us', 'vs', 'ws', 'rprojs',
                                'teffs', 'loggs', 'rs', 'areas',
                                'abs_intensities']
            mesh_quant_do_filt = [False, False, False, False,
                                  False, False, False, False,
                                  True]
            mesh_quant_units = [u.solRad, u.solRad, u.solRad, u.solRad,
                                u.K, 1.0, u.solRad, u.solRad**2,
                                u.W / (u.m**3)]
            
            mesh_quant_filts = []
            for filt in filts_list:
                mesh_quant_filts.append(filt.phoebe_ds_name)
                        
            mesh_quants_pri = {}
            mesh_quants_sec = {}
            
            for (quant, do_filt,
                 quant_unit) in zip(mesh_quant_names, mesh_quant_do_filt,
                                    mesh_quant_units):
                if do_filt:
                    for filt in mesh_quant_filts:
                        quant_pri = b[f'{quant}@primary@{filt}'].value *\
                                    quant_unit
                        quant_sec = b[f'{quant}@secondary@{filt}'].value *\
                                    quant_unit
                    
                        mesh_quants_pri[f'{quant}_{filt}'] = quant_pri
                        mesh_quants_sec[f'{quant}_{filt}'] = quant_sec
                else:
                    quant_pri = b[f'{quant}@primary'].value * quant_unit
                    quant_sec = b[f'{quant}@secondary'].value * quant_unit
                    
                    mesh_quants_pri[quant] = quant_pri
                    mesh_quants_sec[quant] = quant_sec
            
            # Get uvw coordinates of each vertix of triangle in mesh
            uvw_elements_pri = b.get_parameter(qualifier='uvw_elements', 
                                  component='primary', 
                                  dataset='mod_mesh',
                                  kind='mesh', 
                                  context='model').value * u.solRad
            uvw_elements_sec = b.get_parameter(qualifier='uvw_elements', 
                                  component='secondary', 
                                  dataset='mod_mesh',
                                  kind='mesh', 
                                  context='model').value * u.solRad
            
            mesh_quants_pri['v1_us'] = uvw_elements_pri[:,0,0]
            mesh_quants_pri['v1_vs'] = uvw_elements_pri[:,0,1]
            mesh_quants_pri['v1_ws'] = uvw_elements_pri[:,0,2]
            
            mesh_quants_pri['v2_us'] = uvw_elements_pri[:,1,0]
            mesh_quants_pri['v2_vs'] = uvw_elements_pri[:,1,1]
            mesh_quants_pri['v2_ws'] = uvw_elements_pri[:,1,2]
            
            mesh_quants_pri['v3_us'] = uvw_elements_pri[:,2,0]
            mesh_quants_pri['v3_vs'] = uvw_elements_pri[:,2,1]
            mesh_quants_pri['v3_ws'] = uvw_elements_pri[:,2,2]
            
            
            mesh_quants_sec['v1_us'] = uvw_elements_sec[:,0,0]
            mesh_quants_sec['v1_vs'] = uvw_elements_sec[:,0,1]
            mesh_quants_sec['v1_ws'] = uvw_elements_sec[:,0,2]
            
            mesh_quants_sec['v2_us'] = uvw_elements_sec[:,1,0]
            mesh_quants_sec['v2_vs'] = uvw_elements_sec[:,1,1]
            mesh_quants_sec['v2_ws'] = uvw_elements_sec[:,1,2]
            
            mesh_quants_sec['v3_us'] = uvw_elements_sec[:,2,0]
            mesh_quants_sec['v3_vs'] = uvw_elements_sec[:,2,1]
            mesh_quants_sec['v3_ws'] = uvw_elements_sec[:,2,2]
            
            
            # Construct mesh tables for each star and output
            mesh_pri_table = Table(mesh_quants_pri)
            mesh_pri_table.sort(['us'], reverse=True)
            with open('mesh_pri.txt', 'w') as out_file:
                for line in mesh_pri_table.pformat_all():
                    out_file.write(line + '\n')
            mesh_pri_table.write('mesh_pri.h5', format='hdf5',
                                 path='data', serialize_meta=True,
                                 overwrite=True)
            mesh_pri_table.write('mesh_pri.fits', format='fits',
                                 overwrite=True)
            
            mesh_sec_table = Table(mesh_quants_sec)
            mesh_sec_table.sort(['us'], reverse=True)
            with open('mesh_sec.txt', 'w') as out_file:
                for line in mesh_sec_table.pformat_all():
                    out_file.write(line + '\n')
            mesh_sec_table.write('mesh_sec.h5', format='hdf5',
                                 path='data', serialize_meta=True,
                                 overwrite=True)
            mesh_sec_table.write('mesh_sec.fits', format='fits',
                                 overwrite=True)
        else:
            mesh_plot_out = b['mod_mesh@model'].plot(save='./binary_mesh{0}.pdf'.format(suffix_str))
    
    
    # Get fluxes
    filt_model_fluxes = ()
    filt_model_mags = ()
    
    # Go through each filter
    for (filt_index, filt) in enumerate(filts_list):
        cur_filt_model_fluxes =\
            np.array(b[f'fluxes@lc@{filt.phoebe_ds_name}@model'].value) *\
            u.W / (u.m**2.)
        cur_filt_model_mags = -2.5 *\
            np.log10(cur_filt_model_fluxes / filt.flux_ref_filt) + 0.03
        
        filt_model_fluxes = filt_model_fluxes + (cur_filt_model_fluxes, )
        filt_model_mags = filt_model_mags + (cur_filt_model_mags, )
    
    # Get RVs
    model_RVs_pri = np.array(b['rvs@primary@run@rv@model'].value) * u.km / u.s
    model_RVs_sec = np.array(b['rvs@secondary@run@rv@model'].value) * u.km / u.s
    
    
    if print_diagnostics:
        print("\nFlux Checks")
        
        for (filt_index, filt) in enumerate(filts_list):
            print("Fluxes, {0}: {1}".format(
                filt.filter_name,
                filt_model_fluxes[filt_index]))
            print("Mags, {0}: {1}".format(
                filt.filter_name,
                filt_model_mags[filt_index]))
        
        print("\nRV Checks")
        print("RVs, Primary: {0}".format(model_RVs_pri))
        print("RVs, Secondary: {0}".format(model_RVs_sec))
        
    if make_mesh_plots:
        return (filt_model_mags,
                model_RVs_pri, model_RVs_sec,
                mesh_plot_out)
    else:
        return (filt_model_mags,
                model_RVs_pri, model_RVs_sec)
    
def phased_obs(observation_times, binary_period, t0,
               filts_list=[kp_filt, h_filt]):
    """Phase observation times to a given binary period and t0
    """
    num_filts = len(filts_list)
    
    # Read in observation times, and separate photometry from RVs
    filt_MJDs = observation_times[:num_filts]
    rv_MJDs = observation_times[num_filts]
    
    out_phased_obs = ()
    # Phase photometric observation times
    for filt_index in range(num_filts):
        # Phase the current filter's MJDs
        cur_filt_MJDs = filt_MJDs[filt_index]
        cur_filt_phased_days = (((cur_filt_MJDs - t0) %
                                 binary_period.to(u.d).value) /
                                binary_period.to(u.d).value)
        
        # Compute phase sorted inds
        cur_filt_phases_sorted_inds = np.argsort(cur_filt_phased_days)
        
        # Compute model times sorted to phase sorted inds
        cur_filt_model_times = cur_filt_phased_days *\
                               binary_period.to(u.d).value
        cur_filt_model_times = cur_filt_model_times[cur_filt_phases_sorted_inds]
        
        # Append calculated values to output tuple
        out_phased_obs = out_phased_obs + \
                         ((cur_filt_phased_days, cur_filt_phases_sorted_inds,
                           cur_filt_model_times), )
    
    # Phase RV observation times
    rv_phased_days = (((rv_MJDs - t0) % binary_period.to(u.d).value) /
                      binary_period.to(u.d).value)
    
    rv_phases_sorted_inds = np.argsort(rv_phased_days)
    
    rv_model_times = (rv_phased_days) * binary_period.to(u.d).value
    rv_model_times = rv_model_times[rv_phases_sorted_inds]
    
    # Append RV values to output tuple
    out_phased_obs = out_phased_obs + \
                     ((rv_phased_days, rv_phases_sorted_inds,
                       rv_model_times), )
    
    return out_phased_obs

def dist_ext_mag_calc(input_mags, target_dist, filt_exts):
    binary_mags_filts = ()
    
    # Calculate distance modulus
    dist_mod = 5. * np.log10(target_dist / (10. * u.pc))
    
    # Adjust magnitudes in each filter
    for (filt_index, filt_ext) in enumerate(filt_exts):
        # Add mag distance modulus and extinction
         adjusted_mags = input_mags[filt_index] + dist_mod + filt_ext
         binary_mags_filts = binary_mags_filts + (adjusted_mags, )
    
    # Return mags adjusted for target distance and extinction
    return binary_mags_filts
    

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
                     isoc_Ks_ext, filt_exts, ext_alpha,
                     isoc_dist, bin_dist,
                     filts_list=[kp_filt, h_filt],
                     use_blackbody_atm=False,
                     use_compact_object=False,
                     irrad_frac_refl=1.0,
                     make_mesh_plots=False, mesh_temp=False,
                     mesh_temp_cmap=None,
                     plot_name=None,
                     num_triangles=1500,
                     print_diagnostics=False,
                    ):
    # Filter calculations
    num_filts = len(filts_list)
    
    isoc_filt_exts = np.empty(num_filts)
    filt_ext_adjs = np.empty(num_filts)
    
    for filt_index, cur_filt in enumerate(filts_list):
        # Calculate extinctions implied by isochrone extinction
        isoc_filt_exts[filt_index] = cur_filt.calc_isoc_filt_ext(
                                         isoc_Ks_ext, ext_alpha)
        
        # Calculate extinction adjustments
        filt_ext_adjs[filt_index] = (filt_exts[filt_index] -
                                     isoc_filt_exts[filt_index])
    
    # Calculate distance modulus adjustments
    dist_mod_mag_adj = 5. * np.log10(bin_dist / (isoc_dist.to(u.pc)).value)
    
    # Extract stellar parameters from input
    (star1_mass, star1_rad, star1_teff, star1_logg,
     star1_filt_mags, star1_filt_pblums) = star1_params_lcfit
    (star2_mass, star2_rad, star2_teff, star2_logg,
     star2_filt_mags, star2_filt_pblums) = star2_params_lcfit
    
    # Run binary star model to get binary mags
    binary_star_lc_out = binary_star_lc(
        star1_params_lcfit,
        star2_params_lcfit,
        binary_params,
        observation_times,
        filts_list=filts_list,
        use_blackbody_atm=use_blackbody_atm,
        use_compact_object=use_compact_object,
        irrad_frac_refl=irrad_frac_refl,
        make_mesh_plots=make_mesh_plots,
        mesh_temp=mesh_temp,
        mesh_temp_cmap=mesh_temp_cmap,
        plot_name=plot_name,
        num_triangles=num_triangles,
        print_diagnostics=print_diagnostics)
    
    if make_mesh_plots:
        (binary_mags_filts,
         binary_RVs_pri, binary_RVs_sec,
         mesh_plot_out) = binary_star_lc_out
    else:
        (binary_mags_filts,
         binary_RVs_pri, binary_RVs_sec) = binary_star_lc_out
    
    # Test failure of binary mag calculation
    if ((binary_mags_filts[0])[0] == -1.):
        return -np.inf
    
    ## Apply distance modulus and isoc. extinction to binary magnitudes
    binary_mags_filts = dist_ext_mag_calc(
                            binary_mags_filts,
                            isoc_dist,
                            isoc_filt_exts,
                        )
    
    # Go through the mag adjustments, filter by filter
    binary_mags_filts_list = list(binary_mags_filts)
    
    for filt_index, cur_filt in enumerate(filts_list):
        # Apply the extinction difference between model and isochrone values
        # for each filter
        binary_mags_filts_list[filt_index] += filt_ext_adjs[filt_index]
        
        # Apply the distance modulus for difference between isoc. distance
        # and binary distance
        # (Same for each filter)
        binary_mags_filts_list[filt_index] += dist_mod_mag_adj
    
    binary_mags_filts = tuple(binary_mags_filts_list)
    
    # Return final light curve
    if make_mesh_plots:
        return (binary_mags_filts,
                binary_RVs_pri, binary_RVs_sec,
                mesh_plot_out)
    else:
        return (binary_mags_filts,
                binary_RVs_pri, binary_RVs_sec)
