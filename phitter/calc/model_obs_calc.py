#!/usr/bin/env python

# Light curve calculation functions,
# for fitting with PHOEBE
# ---
# Abhimat Gautam

import phoebe
from phoebe import u
from phoebe import c as const
from phoebe.parameters.dataset import rv
from spisea import synthetic
from phitter import filters, observables
from phitter.params import star_params
import numpy as np
import sys
import copy
from astropy.table import Table
from astropy import modeling
from astropy.modeling import Model
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.ticker import MultipleLocator

# Filters for default filter list
kp_filt = filters.nirc2_kp_filt()
h_filt = filters.nirc2_h_filt()

# Stellar Parameters
# stellar_params = (mass, rad, teff, mag_Kp, mag_H, pblum_Kp, pblum_H)




class binary_star_model_obs(object):
    """
    Class to compute the observables for a modeled binary system,
    given stellar parameters and binary parameters.
    
    Parameters
    ----------
    bin_observables : observables
        observables object, with obs_times, obs_filts, and obs_types specified.
    use_blackbody_atm : bool, default=False
        Use blackbody atmosphere instead of default Castelli & Kurucz
        atmosphere. Default: False (i.e.: using a C&K atm by default)
    use_compact_object : bool, default=False
        If true, sets eclipse_method to 'only_horizon' in PHOEBE, which is
        necessary for compact companions without eclipses. Default: False
    
    print_diagnostics : bool, default=False
        Print diagnostic messages, helpful for debugging.
    par_compute : bool, default=False
        Uses parallelization when computing with PHOEBE.
    num_par_processes : int, default=8
        Number of processes to use when parallel computing with PHOEBE.
    """
    def __init__(
            self,
            bin_observables,
            use_blackbody_atm=False,
            use_compact_object=False,
            print_diagnostics=False,
            par_compute=False, num_par_processes=8,
            *args, **kwargs,
    ):
        super().__init__(
            *args, **kwargs,
        )
        
        self.bin_observables = bin_observables
        self.use_blackbody_atm = use_blackbody_atm
        self.use_compact_object = use_compact_object
        
        self.print_diagnostics = print_diagnostics
        
        self.par_compute = par_compute
        self.num_par_processes = num_par_processes
        
        return
    
    def compute_obs(
            self,
            star1_params, star2_params,
            binary_params,
            irrad_frac_refl=1.0,
            num_triangles=1500,
            make_mesh_plots=False, mesh_temp=False, mesh_temp_cmap=None,
            plot_name=None,
    ):
        """
        Function to compute observables with the specified star and binary
        system parameters.
        
        Parameters
        ----------
        star1_params : star_params
            star_params object, with parameters for the primary star.
        star2_params : star_params
            star_params object, with parameters for the secondary star.
        binary_params : binary_params
            binary_params object, with parameters for the binary system
            configuration.
        irrad_frac_refl : float, default=1.0
            Fraction reflectivity for irradiation
        num_triangles : int, default=1500
            Number of triangles to use for PHOEBE's mesh model of each stellar
            atmosphere. For contact system, num_triangles*2 are used for contact
            envelope.
        make_mesh_plots : bool, default=False
            Make a mesh plot of the binary system. Default: False
        plot_name : str, default=None
            Name for the output plots, if making a mesh plot
        
        Returns
        -------
        observables
            observables object returned. Deep copy of input observables object,
            with obs also defined, with modeled values.
        """
        
        if self.par_compute:
            phoebe.mpi_on(nprocs=self.num_par_processes)
        else:
            phoebe.mpi_off()
        
        # Read in the stellar parameters of the binary components
        star1_mass = star1_params.mass
        star1_rad = star1_params.rad
        star1_teff = star1_params.teff
        star1_logg = star1_params.logg
        star1_filt_pblums = star1_params.pblums
        
        star2_mass = star2_params.mass
        star2_rad = star2_params.rad
        star2_teff = star2_params.teff
        star2_logg = star2_params.logg
        star2_filt_pblums = star2_params.pblums
        
        # Read in the parameters of the binary system
        binary_period = binary_params.period
        binary_ecc = binary_params.ecc
        binary_inc = binary_params.inc
        t0 = binary_params.t0
        
        err_out = ((np.array([-1.]), np.array([-1.])),
                   np.array([-1.]), np.array([-1.]))
        if make_mesh_plots:
            err_out = ((np.array([-1.]), np.array([-1.])),
                       np.array([-1.]), np.array([-1.]),
                       np.array([-1.]))
        
        # Check for high temp ck2004 atmosphere limits
        if not self.use_blackbody_atm:
            # log g = 3.5, high temp bounds (above 31e3 K)
            if star1_teff > (31_000 * u.K) and (4.0 > star1_logg > 3.5):
                star1_teff_round = 30995.0 * u.K
                
                if self.print_diagnostics:
                    print('Star 1 out of C&K 2004 grid')
                    print('star1_logg = {0:.4f}'.format(star1_logg))
                    print('Rounding down star1_teff')
                    print('{0:.4f} -> {1:.4f}'.format(star1_teff, star1_teff_round))
                
                star1_teff = star1_teff_round
            if star2_teff > (31_000 * u.K) and (4.0 > star2_logg > 3.5):
                star2_teff_round = 30995.0 * u.K
                
                if self.print_diagnostics:
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
        binary_sma = (
            (binary_period**2. * const.G * (star1_mass + star2_mass)) /\
            (4. * np.pi**2.)
        )**(1./3.)
        
        binary_q = star2_mass / star1_mass
        
        if self.print_diagnostics:
            print('\nBinary orbit checks')
            print('Binary SMA: {0}'.format(binary_sma.to(u.AU)))
            print('Binary Mass Ratio (q): {0}'.format(binary_q))
        
        b.set_value('period@orbit', binary_period)
        b.set_value('sma@binary@component', binary_sma)
        b.set_value('q@binary@component', binary_q)
        
        ## Inclination
        b.set_value('incl@orbit', binary_inc)
        b.set_value('long_an@orbit', 180.*u.deg)
        b.set_value('per0@orbit', 180.*u.deg)
        
        if self.print_diagnostics:
            print("\nBinary orbit characteristics:")
            print(b.filter('binary@orbit'))
            # print(b.get_parameter('t0_perpass@binary@component'))
            # print(b.get_parameter('t0_supconj@binary@component'))
            # print(b.get_parameter('long_an@binary@component'))
            # print(b.get_parameter('per0@binary@component'))
            # print(b.get_parameter('incl@orbit'))
        
        # Check for overflow
        
        # Variables to help store the non-detached binary cases
        star1_semidetached = False
        star2_semidetached = False
        
        star1_overflow = False
        star2_overflow = False
        
        # Get the max radii for both component stars
        star1_rad_max = b.get_value('requiv_max@primary@component') * u.solRad
        star2_rad_max = b.get_value('requiv_max@secondary@component') * u.solRad
        
        ## Check for semidetached cases
        if self.print_diagnostics:
            print('\nSemidetached checks')
            print('Star 1: {0}'.format(np.abs((star1_rad - star1_rad_max) / star1_rad_max)))
            print('Star 2: {0}'.format(np.abs((star2_rad - star2_rad_max) / star2_rad_max)))
        
        semidet_cut = 0.001   # (within 0.1% of max radii)
        semidet_cut = 0.015   # (within 1.5% of max radii)
        
        if np.abs((star1_rad - star1_rad_max) / star1_rad_max) < semidet_cut:
            star1_semidetached = True
        if np.abs((star2_rad - star2_rad_max) / star2_rad_max) < semidet_cut:
            star2_semidetached = True
        
        # Check for overflow
        if (star1_rad > star1_rad_max) and not star1_semidetached:
            star1_overflow = True
        
        if (star2_rad > star2_rad_max) and not star2_semidetached:
            star2_overflow = True
        
        # If both stars are overflowing, check which star overflows more?
        # Only choose that star to be overflowing
        if star1_overflow and star2_overflow:
            if (star1_rad - star1_rad_max) >= (star2_rad - star2_rad_max):
                star2_overflow = False
            else:
                star1_overflow = False
        
        
        if self.print_diagnostics:
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
            redo_binary_params = copy.deepcopy(binary_params)
            redo_binary_params.t0 = t0 - (binary_period.to(u.d).value / 2.)
            
            return self.compute_obs(
                star2_params, star1_params,
                redo_binary_params,
                irrad_frac_refl=irrad_frac_refl,
                num_triangles=num_triangles,
                make_mesh_plots=make_mesh_plots,
                mesh_temp=mesh_temp, mesh_temp_cmap=mesh_temp_cmap,
                plot_name=plot_name,
            )
        
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
        if self.use_blackbody_atm:
            b.add_compute(
                'phoebe', compute='detailed', irrad_method='wilson',
                atm='blackbody',
            )
            
            # b.add_compute('phoebe', compute='detailed',
            #               irrad_method='horvat', atm='blackbody', distortion_method='rotstar')
            
            b.set_value('atm@primary@detailed', 'blackbody')
            b.set_value('atm@secondary@detailed', 'blackbody')
        else:
            b.add_compute(
                'phoebe', compute='detailed', irrad_method='wilson'
            )
        
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
        
        if self.print_diagnostics:
           print('Contact envelope parameters:')
           print(b.filter('contact_envelope@detailed@compute'))
        
        # Set the number of triangles in the mesh
        # Detached stars
        if len(b.filter('ntriangles@primary@detailed@compute')) == 1: 
            b.set_value(
                'ntriangles@primary@detailed@compute',
                num_triangles,
            )
            
        if len(b.filter('ntriangles@secondary@detailed@compute')) == 1: 
            b.set_value(
                'ntriangles@secondary@detailed@compute',
                num_triangles,
            )
        
        # Contact envelope
        if len(b.filter('ntriangles@contact_envelope@detailed@compute')) == 1:
           if self.print_diagnostics:
               print('Setting number of triangles for contact envelope')
           b.set_value(
               'ntriangles@contact_envelope@detailed@compute',
               num_triangles * 2,
           )
        
        # Extract times for modeling photometry
        phot_filt_MJDs = {}     # times, organized by filt
        rv_filt_MJDs = {}       # times, organized by filt
        
        phot_filt_phase = {}
        rv_filt_phase = {}
        
        if self.bin_observables.num_obs_phot > 0:
            for filt in self.bin_observables.unique_filts_phot:
                phot_filt_MJDs[filt] = self.bin_observables.obs_times[
                    self.bin_observables.phot_filt_filters[filt]
                ]
                phot_filt_phase[filt] =\
                    ((phot_filt_MJDs[filt] - t0) %
                     binary_period.to(u.d).value) / binary_period.to(u.d).value
        
        if self.bin_observables.num_obs_rv > 0:
            for filt in self.bin_observables.unique_filts_rv:
                rv_filt_MJDs[filt] = self.bin_observables.obs_times[
                    self.bin_observables.rv_filt_filters[filt]
                ]
                rv_filt_phase[filt] =\
                    ((rv_filt_MJDs[filt] - t0) %
                     binary_period.to(u.d).value) / binary_period.to(u.d).value
        
        # Add light curve datasets    
        if self.bin_observables.num_obs_phot > 0:
            # Parameters to change if using a blackbody atmosphere
            if self.use_blackbody_atm:
                b.set_value_all('ld_mode_bol', 'manual')
                b.set_value_all('ld_func_bol', 'linear')
                b.set_value_all('ld_coeffs_bol', [0.0])
            
            # Set irradiation reflection fraction 
            b.set_value_all('irrad_frac_refl_bol', irrad_frac_refl)
            
            # Change in setup for compact object
            if self.use_compact_object:
                b.set_value('irrad_method@detailed', 'none')
            
            # Go through each filter and add a lightcurve dataset
            for (filt_index, filt) in enumerate(self.bin_observables.unique_filts_phot):
                # PHOEBE sorts outputs by time, so do it ourselves first
                filt_phases_sorted_inds = np.argsort(
                    phot_filt_phase[filt]
                )
                
                filt_model_times = \
                    phot_filt_phase[filt] * binary_period.to(u.d).value
                filt_model_times = filt_model_times[filt_phases_sorted_inds]
                
                b.add_dataset(
                    phoebe.dataset.lc,
                    time=filt_model_times,
                    dataset=filt.phoebe_ds_name,
                    passband=filt.phoebe_pb_name,
                )
                if self.use_blackbody_atm:
                    b.set_value_all('ld_mode@' + filt.phoebe_ds_name, 'manual')
            
                    b.set_value_all('ld_func@' + filt.phoebe_ds_name, 'linear')
            
                    b.set_value_all('ld_coeffs@' + filt.phoebe_ds_name, [0.0])
        
        # Add RV datasets
        if self.bin_observables.num_obs_rv > 0:
            # Go through each filter and add a RV dataset with all RV times
            rv_model_times = np.array([])
            
            for (filt_index, filt) in enumerate(self.bin_observables.unique_filts_rv):
                # PHOEBE sorts outputs by time, so do it ourselves first
                filt_phases_sorted_inds = np.argsort(
                    rv_filt_phase[filt]
                )
                
                filt_model_times = \
                    rv_filt_phase[filt] * binary_period.to(u.d).value
                filt_model_times = filt_model_times[filt_phases_sorted_inds]
                
                rv_model_times = np.append(rv_model_times, filt_model_times)
            
                
            b.add_dataset(
                phoebe.dataset.rv,
                time=rv_model_times,
                dataset='mod_rv',
                passband=(self.bin_observables.unique_filts_rv)[0].phoebe_pb_name,
            )
            if self.use_blackbody_atm:
                b.set_value_all('ld_mode@mod_rv', 'manual')
                
                b.set_value_all('ld_func@mod_rv', 'linear')
                
                b.set_value_all('ld_coeffs@mod_rv', [0.0])
        
        # Add mesh dataset if making mesh plot
        if make_mesh_plots:
            # Only generating mesh at phase of 0.25
            b.add_dataset(
                'mesh',
                times=[(binary_period/4.).to(u.d).value],
                dataset='mod_mesh',
            )
            b.set_value('coordinates@mesh', ['uvw'])
            if mesh_temp:
                mesh_columns = [
                    'us', 'vs', 'ws', 'rprojs',
                    'teffs', 'loggs', 'rs', 'areas',
                ]
                for filt in self.bin_observables.unique_filts_phot:
                    mesh_columns.append('*@' + filt.phoebe_ds_name)
                
                b['columns@mesh'] = mesh_columns
        
        # Set the passband luminosities for the stars
        if (not star1_overflow) and (not star2_overflow):
            # Detached / semidetached case
            for (filt_index, filt) in enumerate(self.bin_observables.unique_filts_phot):
                b.set_value(
                    'pblum_mode@' + filt.phoebe_ds_name,
                    'decoupled',
                )
                
                b.set_value(
                    'pblum@primary@' + filt.phoebe_ds_name,
                    star1_filt_pblums[filt_index],
                )
                
                b.set_value(
                    'pblum@secondary@' + filt.phoebe_ds_name,
                    star2_filt_pblums[filt_index],
                )
        else:
            if star1_overflow:
                for (filt_index, filt) in enumerate(self.bin_observables.unique_filts_phot):
                    b.set_value(
                        'pblum@primary@' + filt.phoebe_ds_name,
                        star1_filt_pblums[filt_index],
                    )
            elif star2_overflow:
                for (filt_index, filt) in enumerate(self.bin_observables.unique_filts_phot):
                    b.set_value(
                        'pblum@secondary@' + filt.phoebe_ds_name,
                        star2_filt_pblums[filt_index],
                    )
        
        # Run compute
        # Determine eclipse method
        if self.use_compact_object:
            eclipse_method = 'only_horizon'
        else:
            eclipse_method = 'native'
        
        if self.print_diagnostics:
            print("Trying inital compute run")
            b.run_compute(
                compute='detailed', model='run',
                progressbar=False, eclipse_method=eclipse_method,
            )
        else:
            try:
                b.run_compute(compute='detailed', model='run',
                              progressbar=False, eclipse_method=eclipse_method)
            except:
                if self.print_diagnostics:
                    print("Error during primary binary compute: {0}".format(
                                sys.exc_info()[0])
                         )
                return err_out
        
        bin_observables_out = copy.deepcopy(self.bin_observables)
        
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
                suffix_str = '_' + self.plot_name
            
            ## Mesh plot
            if mesh_temp:
                mesh_plot_out = b['mod_mesh@model'].plot(
                                    save='./binary_mesh{0}.pdf'.format(suffix_str),
                                    fc='teffs',
                                    fcmap=self.mesh_temp_cmap,
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
                for filt in self.bin_observables.unique_filts_phot:
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
        phot_model_fluxes = {}
        phot_model_mags = {}
        
        # Go through each filter
        for (filt_index, filt) in enumerate(self.bin_observables.unique_filts_phot):
            cur_filt_model_fluxes =\
                np.array(b[f'fluxes@lc@{filt.phoebe_ds_name}@model'].value) *\
                u.W / (u.m**2.)
            cur_filt_model_mags = -2.5 *\
                np.log10(cur_filt_model_fluxes / filt.flux_ref_filt) + 0.03
            
            phot_model_fluxes[filt] = cur_filt_model_fluxes
            phot_model_mags[filt] = cur_filt_model_mags
        
        # Get RVs
        model_RVs_pri = np.array(b['rvs@primary@run@rv@model'].value) * u.km / u.s
        model_RVs_sec = np.array(b['rvs@secondary@run@rv@model'].value) * u.km / u.s
        
        
        if self.print_diagnostics:
            print("\nFlux Checks")
            
            for (filt_index, filt) in enumerate(self.bin_observables.unique_filts_phot):
                print("Fluxes, {0}: {1}".format(
                    filt.filter_name,
                    phot_model_fluxes[filt]))
                print("Mags, {0}: {1}".format(
                    filt.filter_name,
                    phot_model_mags[filt]))
            
            print("\nRV Checks")
            print("RVs, Primary: {0}".format(model_RVs_pri))
            print("RVs, Secondary: {0}".format(model_RVs_sec))
        
        if self.print_diagnostics:
            print("\nFlux Checks")
            
            for (filt_index, filt) in enumerate(self.bin_observables.unique_filts_phot):
                print("Fluxes, {0}: {1}".format(
                    filt.filter_name,
                    phot_model_fluxes[filt]))
                print("Mags, {0}: {1}".format(
                    filt.filter_name,
                    phot_model_mags[filt]))
            
            print("\nRV Checks")
            print("RVs, Primary: {0}".format(model_RVs_pri))
            print("RVs, Secondary: {0}".format(model_RVs_sec))
            
        if make_mesh_plots:
            return (phot_model_mags,
                    model_RVs_pri, model_RVs_sec,
                    mesh_plot_out)
        else:
            return (phot_model_mags,
                    model_RVs_pri, model_RVs_sec)

def single_star_model_obs(
        stellar_params,
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
    

# def binary_star_model_obs(
#         observables, star1_params, star2_params, binary_params,
#         use_blackbody_atm=False,
#         use_compact_object=False,
#         irrad_frac_refl=1.0,
#         num_triangles=1500,
#         make_mesh_plots=False, mesh_temp=False, mesh_temp_cmap=None,
#         plot_name=None,
#         print_diagnostics=False, par_compute=False, num_par_processes=8,
#         
#     ):

# def binary_mags_calc(
#         observables,
#         star1_params, star2_params,
#         binary_params,
#         isoc_Ks_ext, filt_exts, ext_alpha,
#         isoc_dist, bin_dist,
#         use_blackbody_atm=False,
#         use_compact_object=False,
#         irrad_frac_refl=1.0,
#         make_mesh_plots=False, mesh_temp=False,
#         mesh_temp_cmap=None,
#         plot_name=None,
#         par_compute=False, num_par_processes=8,
#         num_triangles=1500,
#         print_diagnostics=False,
#     ):
#     # Filter calculations
#     num_filts = len(filts_list)
#     
#     isoc_filt_exts = np.empty(num_filts)
#     filt_ext_adjs = np.empty(num_filts)
#     
#     for filt_index, cur_filt in enumerate(filts_list):
#         # Calculate extinctions implied by isochrone extinction
#         isoc_filt_exts[filt_index] = cur_filt.calc_isoc_filt_ext(
#                                          isoc_Ks_ext, ext_alpha)
#         
#         if print_diagnostics:
#             print(f'isoc_filt_exts = {isoc_filt_exts}')
#         
#         # Calculate extinction adjustments
#         filt_ext_adjs[filt_index] = (filt_exts[filt_index] -
#                                      isoc_filt_exts[filt_index])
#     
#     # Calculate distance modulus adjustments
#     dist_mod_mag_adj = 5. * np.log10(bin_dist / (isoc_dist.to(u.pc)).value)
#     
#     # Run binary star model to get binary mags
#     binary_star_lc_out = binary_star_model_obs(
#         observables,
#         star1_params,
#         star2_params,
#         binary_params,
#         use_blackbody_atm=use_blackbody_atm,
#         use_compact_object=use_compact_object,
#         num_triangles=num_triangles,
#         irrad_frac_refl=irrad_frac_refl,
#         make_mesh_plots=make_mesh_plots,
#         mesh_temp=mesh_temp,
#         mesh_temp_cmap=mesh_temp_cmap,
#         plot_name=plot_name,
#         par_compute=par_compute, num_par_processes=num_par_processes,
#         print_diagnostics=print_diagnostics,
#     )
#     
#     if make_mesh_plots:
#         (binary_mags_filts,
#          binary_RVs_pri, binary_RVs_sec,
#          mesh_plot_out) = binary_star_lc_out
#     else:
#         (binary_mags_filts,
#          binary_RVs_pri, binary_RVs_sec) = binary_star_lc_out
#     
#     # Test failure of binary mag calculation
#     if ((binary_mags_filts[0])[0] == -1.):
#         return -np.inf
#     
#     ## Apply distance modulus and isoc. extinction to binary magnitudes
#     binary_mags_filts = dist_ext_mag_calc(
#                             binary_mags_filts,
#                             isoc_dist,
#                             isoc_filt_exts,
#                         )
#     
#     # Go through the mag adjustments, filter by filter
#     binary_mags_filts_list = list(binary_mags_filts)
#     
#     for filt_index, cur_filt in enumerate(filts_list):
#         # Apply the extinction difference between model and isochrone values
#         # for each filter
#         binary_mags_filts_list[filt_index] += filt_ext_adjs[filt_index]
#         
#         # Apply the distance modulus for difference between isoc. distance
#         # and binary distance
#         # (Same for each filter)
#         binary_mags_filts_list[filt_index] += dist_mod_mag_adj
#     
#     binary_mags_filts = tuple(binary_mags_filts_list)
#     
#     # Return final light curve
#     if make_mesh_plots:
#         return (binary_mags_filts,
#                 binary_RVs_pri, binary_RVs_sec,
#                 mesh_plot_out)
#     else:
#         return (binary_mags_filts,
#                 binary_RVs_pri, binary_RVs_sec)
