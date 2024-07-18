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
import autofig

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
            make_mesh_plots=False,
            mesh_plot_phases=np.array([0.25]),
            animate=False,
            mesh_plot_fig=None,
            mesh_plot_subplot_grid=None,
            mesh_plot_subplot_grid_indexes=None,
            mesh_temp=False, mesh_temp_cmap=None,
            plot_name=None,
            mesh_plot_kwargs={},
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
        star1_syncpar = star1_params.syncpar
        star1_filt_pblums = star1_params.pblums
        
        star2_mass = star2_params.mass
        star2_rad = star2_params.rad
        star2_teff = star2_params.teff
        star2_logg = star2_params.logg
        star2_syncpar = star2_params.syncpar
        star2_filt_pblums = star2_params.pblums
        
        # Read in the parameters of the binary system
        binary_period = binary_params.period
        binary_ecc = binary_params.ecc
        binary_inc = binary_params.inc
        t0 = binary_params.t0
        
        err_out = observables.observables(
            obs_times=np.array([np.nan]),
            obs=np.array([np.nan]),
        )
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
                mesh_plot_phases=mesh_plot_phases,
                animate=animate,
                mesh_plot_fig=mesh_plot_fig,
                mesh_plot_subplot_grid=mesh_plot_subplot_grid,
                mesh_plot_subplot_grid_indexes=mesh_plot_subplot_grid_indexes,
                mesh_temp=mesh_temp, mesh_temp_cmap=mesh_temp_cmap,
                plot_name=plot_name,
                mesh_plot_kwargs=mesh_plot_kwargs,
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
        
        # Gravity darkening coefficient
        if star1_teff >= 8000 * u.K:    # Radiative
            b.set_value('gravb_bol@primary@component', 1.0)
        elif star1_teff < 6600 * u.K:   # Convective
            b.set_value('gravb_bol@primary@component', 0.32)
        else:
            # Interpolate between the radiative and convective cases
            star1_gravb = ((star1_teff.to(u.K).value - 6600) / (8000 - 6600) *\
                (1-0.32)) + 0.32
            b.set_value('gravb_bol@primary@component', star1_gravb)
        
        # syncpar
        # Defined as the ratio between the (sidereal) orbital and
        # the rotational period (wrt the sky).
        # In default binary system, syncpar = 1
        if star1_syncpar != 1.0:
            b.set_value('syncpar@primary@component', star1_syncpar)
        
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
        
        # Gravity darkening coefficient
        if star2_teff >= 8000 * u.K:    # Radiative
            b.set_value('gravb_bol@secondary@component', 1.0)
        elif star2_teff < 6600 * u.K:   # Convective
            b.set_value('gravb_bol@secondary@component', 0.32)
        else:
            # Interpolate between the radiative and convective cases
            star2_gravb = ((star2_teff.to(u.K).value - 6600) / (8000 - 6600) *\
                (1-0.32)) + 0.32
            b.set_value('gravb_bol@secondary@component', star2_gravb)
        
        # syncpar
        # Defined as the ratio between the (sidereal) orbital and
        # the rotational period (wrt the sky).
        # In default binary system, syncpar = 1
        if star2_syncpar != 1.0:
            b.set_value('syncpar@secondary@component', star2_syncpar)
        
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
                    times=filt_model_times,
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
                
                # Only add unique times (e.g. if pri and sec RVs are modeled for
                # a given time)
                filt_model_times = np.unique(filt_model_times)
                
                rv_model_times = np.append(rv_model_times, filt_model_times)
            
                
            b.add_dataset(
                phoebe.dataset.rv,
                times=rv_model_times,
                dataset='mod_rv',
                passband=(self.bin_observables.unique_filts_rv)[0].phoebe_pb_name,
            )
            if self.use_blackbody_atm:
                b.set_value_all('ld_mode@mod_rv', 'manual')
                
                b.set_value_all('ld_func@mod_rv', 'linear')
                
                b.set_value_all('ld_coeffs@mod_rv', [0.0])
        
        # Add mesh dataset if making mesh plot
        mesh_plot_times = mesh_plot_phases * binary_period.to(u.d).value
        if make_mesh_plots:
            b.add_dataset(
                'mesh',
                compute_times=mesh_plot_times,
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
                    star1_filt_pblums[filt],
                )
                
                b.set_value(
                    'pblum@secondary@' + filt.phoebe_ds_name,
                    star2_filt_pblums[filt],
                )
        else:
            if star1_overflow:
                for (filt_index, filt) in enumerate(self.bin_observables.unique_filts_phot):
                    b.set_value(
                        'pblum@primary@' + filt.phoebe_ds_name,
                        star1_filt_pblums[filt],
                    )
            elif star2_overflow:
                for (filt_index, filt) in enumerate(self.bin_observables.unique_filts_phot):
                    b.set_value(
                        'pblum@secondary@' + filt.phoebe_ds_name,
                        star2_filt_pblums[filt],
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
        
        # Save out mesh plot
        mesh_plot_out = []
        if make_mesh_plots:
            ## Plot Nerdery
            plt.rc('font', family='serif')
            plt.rc('font', serif='Computer Modern Roman')
            plt.rc('text', usetex=True)
            plt.rc('text.latex', preamble=r"\usepackage{gensymb}")
            
            plt.rc('xtick', direction = 'out')
            plt.rc('ytick', direction = 'out')
            # plt.rc('xtick', top = True)
            # plt.rc('ytick', right = True)
            
            plot_name_suffix = ''
            if plot_name is not None:
                plot_name_suffix = f'_{plot_name}'
            
            if animate:
                # Save out animation
                b['mod_mesh@model'].plot(
                    animate=True,
                    save='./binary_mesh{0}.gif'.format(plot_name_suffix),
                    save_kwargs={'writer': 'imagemagick'},
                )
                        
            calls_list = []
            
            # Go through each mesh model time and save
            for mesh_index, mesh_plot_time in enumerate(mesh_plot_times):
                plot_phase_suffix = '_{0}'.format(
                    f'{mesh_plot_phases[mesh_index]:.3f}'.replace('.', 'p')
                )
                
                additional_kwargs = {}
                
                # Add kwargs for handling figure of mesh plot
                if mesh_plot_fig is not None:
                    additional_kwargs['fig'] = mesh_plot_fig
                else:
                    additional_kwargs['save'] =\
                        './binary_mesh{0}{1}.pdf'.format(
                            plot_name_suffix, plot_phase_suffix,
                        )
                
                # Add kwargs if coloring mesh plot by Teff
                if mesh_temp:
                    additional_kwargs['fc'] = 'teffs'
                    additional_kwargs['fcmap'] = mesh_temp_cmap
                    additional_kwargs['ec'] = 'face'
                
                # Add kwargs if making grid of mesh plots with a subplot grid
                if (mesh_plot_subplot_grid is not None and
                    mesh_plot_subplot_grid_indexes is not None):
                    additional_kwargs['axpos'] = (
                        mesh_plot_subplot_grid[0],
                        mesh_plot_subplot_grid[1],
                        mesh_plot_subplot_grid_indexes[mesh_index],
                    )
                
                # Draw plot
                (mesh_af_fig, mesh_plt_fig) = b['mod_mesh@model'].plot(
                    time=mesh_plot_time,
                    **additional_kwargs,
                    **mesh_plot_kwargs,
                )
                
                mesh_plot_out.append(mesh_plt_fig)
            
            if mesh_plot_fig is not None:
                additional_kwargs = {}
                
                # Have to turn off sidebar if making subplots with Teff faces
                if mesh_temp and mesh_plot_subplot_grid is not None:
                    additional_kwargs['draw_sidebars'] = False
                
                (mesh_af_fig, mesh_plt_fig) = b['mod_mesh@model'].show(
                    save='./binary_mesh{0}{1}.pdf'.format(
                        plot_name_suffix, plot_phase_suffix,
                    ),
                    show=False,
                    **additional_kwargs,
                    **mesh_plot_kwargs,
                )
            
            # mesh_plot_fig = mesh_plt_fig
            
            ## Mesh plot
            # if mesh_temp:
            #     mesh_plot_out = b['mod_mesh@model'].plot(
            #                         save='./binary_mesh{0}.pdf'.format(suffix_str),
            #                         fc='teffs',
            #                         fcmap=self.mesh_temp_cmap,
            #                         ec='none',
            #                     )
            #                                              
            #     # print(mesh_plot_out.axs)
            #     
            #     # Extract and output mesh quantities
            #     mesh_quant_names = ['us', 'vs', 'ws', 'rprojs',
            #                         'teffs', 'loggs', 'rs', 'areas',
            #                         'abs_intensities']
            #     mesh_quant_do_filt = [False, False, False, False,
            #                           False, False, False, False,
            #                           True]
            #     mesh_quant_units = [u.solRad, u.solRad, u.solRad, u.solRad,
            #                         u.K, 1.0, u.solRad, u.solRad**2,
            #                         u.W / (u.m**3)]
            #     
            #     mesh_quant_filts = []
            #     for filt in self.bin_observables.unique_filts_phot:
            #         mesh_quant_filts.append(filt.phoebe_ds_name)
            #                 
            #     mesh_quants_pri = {}
            #     mesh_quants_sec = {}
            #     
            #     for (quant, do_filt,
            #          quant_unit) in zip(mesh_quant_names, mesh_quant_do_filt,
            #                             mesh_quant_units):
            #         if do_filt:
            #             for filt in mesh_quant_filts:
            #                 quant_pri = b[f'{quant}@primary@{filt}'].value *\
            #                             quant_unit
            #                 quant_sec = b[f'{quant}@secondary@{filt}'].value *\
            #                             quant_unit
            #             
            #                 mesh_quants_pri[f'{quant}_{filt}'] = quant_pri
            #                 mesh_quants_sec[f'{quant}_{filt}'] = quant_sec
            #         else:
            #             quant_pri = b[f'{quant}@primary'].value * quant_unit
            #             quant_sec = b[f'{quant}@secondary'].value * quant_unit
            #             
            #             mesh_quants_pri[quant] = quant_pri
            #             mesh_quants_sec[quant] = quant_sec
            #     
            #     # Get uvw coordinates of each vertix of triangle in mesh
            #     uvw_elements_pri = b.get_parameter(qualifier='uvw_elements', 
            #                           component='primary', 
            #                           dataset='mod_mesh',
            #                           kind='mesh', 
            #                           context='model').value * u.solRad
            #     uvw_elements_sec = b.get_parameter(qualifier='uvw_elements', 
            #                           component='secondary', 
            #                           dataset='mod_mesh',
            #                           kind='mesh', 
            #                           context='model').value * u.solRad
            #     
            #     mesh_quants_pri['v1_us'] = uvw_elements_pri[:,0,0]
            #     mesh_quants_pri['v1_vs'] = uvw_elements_pri[:,0,1]
            #     mesh_quants_pri['v1_ws'] = uvw_elements_pri[:,0,2]
            #     
            #     mesh_quants_pri['v2_us'] = uvw_elements_pri[:,1,0]
            #     mesh_quants_pri['v2_vs'] = uvw_elements_pri[:,1,1]
            #     mesh_quants_pri['v2_ws'] = uvw_elements_pri[:,1,2]
            #     
            #     mesh_quants_pri['v3_us'] = uvw_elements_pri[:,2,0]
            #     mesh_quants_pri['v3_vs'] = uvw_elements_pri[:,2,1]
            #     mesh_quants_pri['v3_ws'] = uvw_elements_pri[:,2,2]
            #     
            #     
            #     mesh_quants_sec['v1_us'] = uvw_elements_sec[:,0,0]
            #     mesh_quants_sec['v1_vs'] = uvw_elements_sec[:,0,1]
            #     mesh_quants_sec['v1_ws'] = uvw_elements_sec[:,0,2]
            #     
            #     mesh_quants_sec['v2_us'] = uvw_elements_sec[:,1,0]
            #     mesh_quants_sec['v2_vs'] = uvw_elements_sec[:,1,1]
            #     mesh_quants_sec['v2_ws'] = uvw_elements_sec[:,1,2]
            #     
            #     mesh_quants_sec['v3_us'] = uvw_elements_sec[:,2,0]
            #     mesh_quants_sec['v3_vs'] = uvw_elements_sec[:,2,1]
            #     mesh_quants_sec['v3_ws'] = uvw_elements_sec[:,2,2]
            #     
            #     
            #     # Construct mesh tables for each star and output
            #     mesh_pri_table = Table(mesh_quants_pri)
            #     mesh_pri_table.sort(['us'], reverse=True)
            #     with open('mesh_pri.txt', 'w') as out_file:
            #         for line in mesh_pri_table.pformat_all():
            #             out_file.write(line + '\n')
            #     mesh_pri_table.write('mesh_pri.h5', format='hdf5',
            #                          path='data', serialize_meta=True,
            #                          overwrite=True)
            #     mesh_pri_table.write('mesh_pri.fits', format='fits',
            #                          overwrite=True)
            #     
            #     mesh_sec_table = Table(mesh_quants_sec)
            #     mesh_sec_table.sort(['us'], reverse=True)
            #     with open('mesh_sec.txt', 'w') as out_file:
            #         for line in mesh_sec_table.pformat_all():
            #             out_file.write(line + '\n')
            #     mesh_sec_table.write('mesh_sec.h5', format='hdf5',
            #                          path='data', serialize_meta=True,
            #                          overwrite=True)
            #     mesh_sec_table.write('mesh_sec.fits', format='fits',
            #                          overwrite=True)
            # else:
            #     mesh_plot_out = b['mod_mesh@model'].plot(save='./binary_mesh{0}.pdf'.format(suffix_str))
        
        
        # Get fluxes
        phot_model_fluxes = {}
        phot_model_mags = {}
        
        if self.bin_observables.num_obs_phot > 0:        
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
        if self.bin_observables.num_obs_rv > 0:
            model_RVs_pri = np.array(b['rvs@primary@run@rv@model'].value)
            model_RVs_sec = np.array(b['rvs@secondary@run@rv@model'].value)
        else:
            model_RVs_pri = np.array([])
            model_RVs_sec = np.array([])
        
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
        
        # Set up output observables object
        bin_observables_out = copy.deepcopy(self.bin_observables)
    
        bin_observables_out.set_obs(
            np.empty(len(bin_observables_out.obs_times))
        )
        
        # Set photometric modeled observations back in original input order
        if bin_observables_out.num_obs_phot > 0:
            for (filt_index, filt) in enumerate(self.bin_observables.unique_filts_phot):
                # Set up filter for observations in correct passband
                obs_filt_filter = np.where(np.logical_and(
                    bin_observables_out.obs_types == 'phot',
                    bin_observables_out.obs_filts == filt,
                ))
                
                modeled_fluxes_filt = bin_observables_out.obs[obs_filt_filter]
                
                # Determine how the original observations were sorted
                filt_phases_sorted_inds = np.argsort(
                    phot_filt_phase[filt]
                )
                
                # Go through each model observation and put that thing back
                # where it came from (, or so help me!)
                for obs_index, sorted_index in enumerate(filt_phases_sorted_inds):
                    modeled_fluxes_filt[obs_index] = phot_model_mags[filt][sorted_index]
                
                # Save back out into output observables object
                bin_observables_out.obs[obs_filt_filter] = modeled_fluxes_filt
                
        # Set RV modeled observations back in original input order
        if bin_observables_out.num_obs_rv > 0:
            for (filt_index, filt) in enumerate(self.bin_observables.unique_filts_rv):
                # Set up filter for observations for correct star
                modeled_RVs_pri = bin_observables_out.obs[
                    bin_observables_out.obs_rv_pri_filter
                ]
                modeled_RVs_sec = bin_observables_out.obs[
                    bin_observables_out.obs_rv_sec_filter
                ]
                
                rv_pri_phases = ((bin_observables_out.obs_times_rv_pri - t0) %
                    binary_period.to(u.d).value) / binary_period.to(u.d).value
                
                rv_sec_phases = ((bin_observables_out.obs_times_rv_pri - t0) %
                    binary_period.to(u.d).value) / binary_period.to(u.d).value
                
                # Get phases of the times that were modeled in model order
                rv_filt_phase[filt] =\
                    ((rv_filt_MJDs[filt] - t0) %
                    binary_period.to(u.d).value) / binary_period.to(u.d).value
                
                # Determine the original observation times
                RV_phases_sorted = np.sort(
                    np.unique(rv_filt_phase[filt])
                )
                
                # Go through each model observation and put that thing back
                # where it came from (, or so help me!)
                for cur_model_index, cur_phase in enumerate(
                    np.unique(RV_phases_sorted)
                ):
                    if cur_phase in rv_pri_phases:
                        modeled_RVs_pri[np.argwhere(rv_pri_phases == cur_phase)] =\
                            model_RVs_pri[cur_model_index]
                    if cur_phase in rv_sec_phases:
                        modeled_RVs_sec[np.argwhere(rv_sec_phases == cur_phase)] =\
                            model_RVs_sec[cur_model_index]
                
                # Save back out into output observables object
                bin_observables_out.obs[
                    bin_observables_out.obs_rv_pri_filter
                ] = modeled_RVs_pri
                bin_observables_out.obs[
                    bin_observables_out.obs_rv_sec_filter
                ] = modeled_RVs_sec
        
        # Return modeled observables, and mesh plot if being made
        if mesh_plot_fig is not None:
            return bin_observables_out, mesh_plot_out, mesh_plot_fig
        elif make_mesh_plots:
            return bin_observables_out, mesh_plot_out
        else:
            return bin_observables_out

class single_star_model_obs(object):
    """
    Class to compute the observables for a modeled binary system,
    given stellar parameters and binary parameters.
    
    Parameters
    ----------
    sing_star_observables : observables
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
            sing_star_observables,
            use_blackbody_atm=False,
            use_compact_object=False,
            print_diagnostics=False,
            par_compute=False, num_par_processes=8,
            *args, **kwargs,
    ):
        super().__init__(
            *args, **kwargs,
        )
        
        self.sing_star_observables = sing_star_observables
        self.use_blackbody_atm = use_blackbody_atm
        self.use_compact_object = use_compact_object
        
        self.print_diagnostics = print_diagnostics
        
        self.par_compute = par_compute
        self.num_par_processes = num_par_processes
        
        return
    
    def compute_obs(
            self,
            star1_params,
            num_triangles=1500,
            make_mesh_plots=False,
            mesh_plot_fig=None,
            mesh_plot_subplot_grid=None,
            mesh_plot_subplot_grid_indexes=None,
            mesh_temp=False, mesh_temp_cmap=None,
            plot_name=None,
            mesh_plot_kwargs={},
    ):
        """
        Function to compute observables with the specified star and binary
        system parameters.
        
        Parameters
        ----------
        star1_params : star_params
            star_params object, with parameters for the star.
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
        star_mass = star1_params.mass
        star_rad = star1_params.rad
        star_teff = star1_params.teff
        star_logg = star1_params.logg
        star_filt_pblums = star1_params.pblums
        
        err_out = observables.observables(
            obs_times=np.array([np.nan]),
            obs=np.array([np.nan]),
        )
        
        if make_mesh_plots:
            err_out = ((np.array([-1.]), np.array([-1.])),
                       np.array([-1.]), np.array([-1.]),
                       np.array([-1.]))
        
        # Check for high temp ck2004 atmosphere limits
        if not self.use_blackbody_atm:
            # log g = 3.5, high temp bounds (above 31e3 K)
            if star_teff > (31_000 * u.K) and (4.0 > star_logg > 3.5):
                star_teff_round = 30995.0 * u.K
                
                if self.print_diagnostics:
                    print('Star is out of C&K 2004 grid')
                    print('star1_logg = {0:.4f}'.format(star_logg))
                    print('Rounding down star1_teff')
                    print('{0:.4f} -> {1:.4f}'.format(star_teff, star_teff_round))
                
                star_teff = star_teff_round
            
            # log g = 4.0, high temp bounds (above 40e3 K)
            if star_teff > (40000 * u.K) and (4.5 > star_logg > 4.0):
                star_teff_round = 39995.0 * u.K
                
                print('Star 1 out of C&K 2004 grid')
                print('star_logg = {0:.4f}'.format(star_logg))
                print('Rounding down star_teff')
                print('{0:.4f} -> {1:.4f}'.format(star_teff, star_teff_round))
                
                star_teff = star_teff_round        
        
        # Set up a single star model
        sing_star = phoebe.default_star()
        
        ## Set a default distance
        sing_star.set_value('distance', 10 * u.pc)
        
        # Set up compute
        if self.use_blackbody_atm:
            sing_star.add_compute(
                'phoebe', compute='detailed',
                distortion_method='sphere',
                irrad_method='none',
                atm='blackbody',
            )
        else:
            sing_star.add_compute(
                'phoebe', compute='detailed',
                distortion_method='sphere',
                irrad_method='none',
            )
        
        # Set the stellar parameters
        sing_star.set_value('teff@component', star_teff)
        sing_star.set_value('mass@component', star_mass)
        # sing_star.set_value('logg@component', star_logg)
        sing_star.set_value('requiv@component', star_rad)
        
        # Setting long rotation period so that large stellar radii can be stable
        sing_star.set_value('period@component', 9e3 * u.yr)
        
        # Gravity darkening coefficient
        if star_teff >= 8000 * u.K:    # Radiative
            sing_star.set_value('gravb_bol@component', 1.0)
        elif star_teff < 6600 * u.K:   # Convective
            sing_star.set_value('gravb_bol@component', 0.32)
        else:
            # Interpolate between the radiative and convective cases
            star_gravb = ((star_teff.to(u.K).value - 6600) / (8000 - 6600) *\
                (1-0.32)) + 0.32
            sing_star.set_value('gravb_bol@component', star_gravb)
        
        # Set the number of triangles in the mesh
        sing_star.set_value('ntriangles@detailed@compute', num_triangles)
        
        # Add light curve datasets    
        if self.sing_star_observables.num_obs_phot > 0:
            # Parameters to change if using a blackbody atmosphere
            if self.use_blackbody_atm:
                sing_star.set_value_all('ld_mode_bol', 'manual')
                sing_star.set_value_all('ld_func_bol', 'linear')
                sing_star.set_value_all('ld_coeffs_bol', [0.0])
            
            # Change in setup for compact object
            if self.use_compact_object:
                sing_star.set_value('irrad_method@detailed', 'none')
            
            # Go through each filter and add a lightcurve dataset
            for (filt_index, filt) in enumerate(self.sing_star_observables.unique_filts_phot):
                sing_star.add_dataset(
                    phoebe.dataset.lc,
                    times=[0],
                    dataset=filt.phoebe_ds_name,
                    passband=filt.phoebe_pb_name,
                )
                if self.use_blackbody_atm:
                    sing_star.set_value_all('ld_mode@' + filt.phoebe_ds_name, 'manual')
                    sing_star.set_value_all('ld_func@' + filt.phoebe_ds_name, 'linear')
            
                    sing_star.set_value_all('ld_coeffs@' + filt.phoebe_ds_name, [0.0])
        
        # Not adding RV datasets since just a single star.
        
        if make_mesh_plots:
            sing_star.add_dataset(
                'mesh',
                compute_times=[0],
                dataset='mod_mesh',
            )
            sing_star.set_value('coordinates@mesh', ['uvw'])
            if mesh_temp:
                mesh_columns = [
                    'us', 'vs', 'ws', 'rprojs',
                    'teffs', 'loggs', 'rs', 'areas',
                ]
                for filt in self.bin_observables.unique_filts_phot:
                    mesh_columns.append('*@' + filt.phoebe_ds_name)
                
                sing_star['columns@mesh'] = mesh_columns
        
        # Set the passband luminosities for the stars
        for (filt_index, filt) in enumerate(self.sing_star_observables.unique_filts_phot):
            sing_star.set_value(
                'pblum_mode@' + filt.phoebe_ds_name,
                'decoupled',
            )
            
            sing_star.set_value(
                'pblum@' + filt.phoebe_ds_name,
                star_filt_pblums[filt],
            )
        
        if self.print_diagnostics:
            print('Star Parameters in PHOEBE:')
            print(sing_star.filter('starA@component'))
        
        # Run compute
        # Determine eclipse method
        if self.use_compact_object:
            eclipse_method = 'only_horizon'
        else:
            eclipse_method = 'native'
        
        if self.print_diagnostics:
            print("Trying inital compute run")
            sing_star.run_compute(
                compute='detailed', model='run',
                progressbar=False,
                # eclipse_method=eclipse_method,
            )
        else:
            try:
                sing_star.run_compute(
                    compute='detailed', model='run',
                    progressbar=False, eclipse_method=eclipse_method,
                )
            except:
                if self.print_diagnostics:
                    print(
                        "Error during primary binary compute: {0}".format(
                            sys.exc_info()[0]
                        )
                    )
                return err_out
        
        # Save out mesh plot
        mesh_plot_out = []
        if make_mesh_plots:
            ## Plot Nerdery
            plt.rc('font', family='serif')
            plt.rc('font', serif='Computer Modern Roman')
            plt.rc('text', usetex=True)
            plt.rc('text.latex', preamble=r"\usepackage{gensymb}")
            
            plt.rc('xtick', direction = 'out')
            plt.rc('ytick', direction = 'out')
            # plt.rc('xtick', top = True)
            # plt.rc('ytick', right = True)
            
            plot_name_suffix = ''
            if plot_name is not None:
                plot_name_suffix = f'_{plot_name}'
            
            calls_list = []
            
            # Go through each mesh model time and save
            for mesh_index, mesh_plot_time in enumerate([0]):
                plot_phase_suffix = ''
                
                additional_kwargs = {}
                
                # Add kwargs for handling figure of mesh plot
                if mesh_plot_fig is not None:
                    additional_kwargs['fig'] = mesh_plot_fig
                else:
                    additional_kwargs['save'] =\
                        './binary_mesh{0}{1}.pdf'.format(
                            plot_name_suffix, plot_phase_suffix,
                        )
                
                # Add kwargs if coloring mesh plot by Teff
                if mesh_temp:
                    additional_kwargs['fc'] = 'teffs'
                    additional_kwargs['fcmap'] = mesh_temp_cmap
                    additional_kwargs['ec'] = 'face'
                
                # Add kwargs if making grid of mesh plots with a subplot grid
                if (mesh_plot_subplot_grid is not None and
                    mesh_plot_subplot_grid_indexes is not None):
                    additional_kwargs['axpos'] = (
                        mesh_plot_subplot_grid[0],
                        mesh_plot_subplot_grid[1],
                        mesh_plot_subplot_grid_indexes[mesh_index],
                    )
                
                # Draw plot
                (mesh_af_fig, mesh_plt_fig) = sing_star['mod_mesh@model'].plot(
                    time=mesh_plot_time,
                    **additional_kwargs,
                    **mesh_plot_kwargs,
                )
                
                mesh_plot_out.append(mesh_plt_fig)
            
            if mesh_plot_fig is not None:
                additional_kwargs = {}
                
                # Have to turn off sidebar if making subplots with Teff faces
                if mesh_temp and mesh_plot_subplot_grid is not None:
                    additional_kwargs['draw_sidebars'] = False
                
                (mesh_af_fig, mesh_plt_fig) = sing_star['mod_mesh@model'].show(
                    save='./binary_mesh{0}{1}.pdf'.format(
                        plot_name_suffix, plot_phase_suffix,
                    ),
                    show=False,
                    **additional_kwargs,
                    **mesh_plot_kwargs,
                )
        
        # Get fluxes
        phot_model_fluxes = {}
        phot_model_mags = {}
        
        if self.sing_star_observables.num_obs_phot > 0:        
            # Go through each filter
            for (filt_index, filt) in enumerate(self.sing_star_observables.unique_filts_phot):
                cur_filt_model_fluxes =\
                    np.array(sing_star[f'fluxes@lc@{filt.phoebe_ds_name}@model'].value) *\
                    u.W / (u.m**2.)
                cur_filt_model_mags = -2.5 *\
                    np.log10(cur_filt_model_fluxes / filt.flux_ref_filt) + 0.03
                
                phot_model_fluxes[filt] = cur_filt_model_fluxes
                phot_model_mags[filt] = cur_filt_model_mags
        
        # Get RVs
        if self.sing_star_observables.num_obs_rv > 0:
            model_RVs_pri = np.array([0.])
            model_RVs_sec = np.array([0.])
        else:
            model_RVs_pri = np.array([])
            model_RVs_sec = np.array([])
        
        if self.print_diagnostics:
            print("\nFlux Checks")
            
            for (filt_index, filt) in enumerate(self.sing_star_observables.unique_filts_phot):
                print("Fluxes, {0}: {1}".format(
                    filt.filter_name,
                    phot_model_fluxes[filt]))
                print("Mags, {0}: {1}".format(
                    filt.filter_name,
                    phot_model_mags[filt]))
            
            print("\nRV Checks")
            print("RVs, Primary: {0}".format(model_RVs_pri))
            print("RVs, Secondary: {0}".format(model_RVs_sec))
        
        # Set up output observables object
        sing_star_observables_out = copy.deepcopy(self.sing_star_observables)
        
        sing_star_observables_out.set_obs(
            np.empty(len(sing_star_observables_out.obs_times))
        )
        
        # Set photometric modeled observations back in original input order
        if sing_star_observables_out.num_obs_phot > 0:
            for (filt_index, filt) in enumerate(self.sing_star_observables.unique_filts_phot):
                # Set up filter for observations in correct passband
                obs_filt_filter = np.where(np.logical_and(
                    sing_star_observables_out.obs_types == 'phot',
                    sing_star_observables_out.obs_filts == filt,
                ))
                
                sing_star_observables_out.obs[
                    obs_filt_filter
                ] = phot_model_mags[filt]
        
        # Set RV modeled observations back in original input order
        if sing_star_observables_out.num_obs_rv > 0:
            for (filt_index, filt) in enumerate(self.sing_star_observables.unique_filts_rv):
                # Set up filter for observations for correct star
                sing_star_observables_out.obs[
                    sing_star_observables_out.obs_rv_pri_filter
                ] = model_RVs_pri
                sing_star_observables_out.obs[
                    sing_star_observables_out.obs_rv_sec_filter
                ] = model_RVs_sec
                
        # Return modeled observables, and mesh plot if being made
        if mesh_plot_fig is not None:
            return sing_star_observables_out, mesh_plot_out, mesh_plot_fig
        elif make_mesh_plots:
            return sing_star_observables_out, mesh_plot_out
        else:
            return sing_star_observables_out
