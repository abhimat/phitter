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

from phoebe_phitter import lc_calc, isoc_interp

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.ticker import MultipleLocator

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

class mcmc_fitter_rad_interp(object):
    # Filter properties
    lambda_Ks = 2.18e-6 * u.m
    dlambda_Ks = 0.35e-6 * u.m

    lambda_Kp = 2.124e-6 * u.m
    dlambda_Kp = 0.351e-6 * u.m

    lambda_H = 1.633e-6 * u.m
    dlambda_H = 0.296e-6 * u.m
    
    ks_filt_info = synthetic.get_filter_info('naco,Ks')
    kp_filt_info = synthetic.get_filter_info('nirc2,Kp')
    h_filt_info = synthetic.get_filter_info('nirc2,H')

    flux_ref_Ks = ks_filt_info.flux0 * (u.erg / u.s) / (u.cm**2.)
    flux_ref_Kp = kp_filt_info.flux0 * (u.erg / u.s) / (u.cm**2.)
    flux_ref_H = h_filt_info.flux0 * (u.erg / u.s) / (u.cm**2.)
    
    # Extinction law (using Nogueras-Lara+ 2018)
    ext_alpha = 2.30
    
    # Number of triangles and atmosphere to use for binary model
    use_blackbody_atm = False
    model_numTriangles = 1500
    
    # Model H Extinction Modifier
    default_H_ext_mod = 0.0
    model_H_ext_mod = True
    
    # Model eccentricity
    default_ecc = 0.0
    model_eccentricity = True
    
    # Model distance
    default_dist = 7.971e3
    model_distance = True
    
    # Default prior bounds
    lo_Kp_ext_prior_bound = 2.0
    hi_Kp_ext_prior_bound = 4.0
    
    lo_H_ext_mod_prior_bound = -2.0
    hi_H_ext_mod_prior_bound = 2.0
    
    lo_inc_prior_bound = 0.
    hi_inc_prior_bound = 180.
    
    lo_period_prior_bound = 79.
    hi_period_prior_bound = 81.
    
    lo_ecc_prior_bound = -0.1
    hi_ecc_prior_bound = 0.1
    
    lo_dist_prior_bound = 7600.
    hi_dist_prior_bound = 8200.
    
    lo_t0_prior_bound = 51773.0
    hi_t0_prior_bound = 51774.0
    
    def __init__(self):
        return
    
    # Function to make and store isochrone
    def make_isochrone(self, age, Ks_ext, dist, phase, met, use_atm_func='merged'):
        self.Ks_ext = Ks_ext
        
        self.dist = dist*u.pc
        self.default_dist = dist
        ## Revise prior bounds for distance
        self.lo_dist_prior_bound = 0.8 * dist
        self.hi_dist_prior_bound = 1.2 * dist
        
        self.age = age
        self.met = met
        
        self.isochrone = isoc_interp.isochrone_mist(age=age,
                             ext=Ks_ext, dist=dist, phase=phase, met=met,
                             use_atm_func=use_atm_func)
        
        ## Convert from specified extinction in Ks to Kp and H
        self.Kp_ext = Ks_ext * (self.lambda_Ks / self.lambda_Kp)**self.ext_alpha
        self.H_ext = Ks_ext * (self.lambda_Ks / self.lambda_H)**self.ext_alpha
    
    # Function to set observation times
    def set_observation_times(self, Kp_observation_times, H_observation_times):
        self.Kp_observation_times = Kp_observation_times
        self.H_observation_times = H_observation_times
        
        self.observation_times = (self.Kp_observation_times, self.H_observation_times)
    
    # Function to set observation mags
    def set_observation_mags(self, kp_obs_mags, kp_obs_mag_errors,
            h_obs_mags, h_obs_mag_errors):
        self.kp_obs_mags = kp_obs_mags
        self.h_obs_mags = h_obs_mags
        
        self.kp_obs_mag_errors = kp_obs_mag_errors
        self.h_obs_mag_errors = h_obs_mag_errors
    
    # Function to set model mesh number of triangles
    def set_model_numTriangles(self, model_numTriangles):
        self.model_numTriangles = model_numTriangles
    
    # Function to set if using blackbody atmosphere
    def set_model_use_blackbody_atm(self, use_blackbody_atm):
        self.use_blackbody_atm = use_blackbody_atm
    
    # Function to set for modelling H extinction modifier
    def set_model_H_ext_mod(self, model_H_ext_mod):
        self.model_H_ext_mod = model_H_ext_mod
    
    # Function to set for modelling eccentricity
    def set_model_eccentricity(self, model_eccentricity):
        self.model_eccentricity = model_eccentricity
    
    # Function to set for modelling distance
    def set_model_distance(self, model_distance):
        self.model_distance = model_distance
    
    # Functions to define prior bounds
    def set_Kp_ext_prior_bounds(self, lo_bound, hi_bound):
        self.lo_Kp_ext_prior_bound = lo_bound
        self.hi_Kp_ext_prior_bound = hi_bound
    
    def set_H_ext_mod_prior_bounds(self, lo_bound, hi_bound):
        self.lo_H_ext_mod_prior_bound = lo_bound
        self.hi_H_ext_mod_prior_bound = hi_bound
    
    def set_inc_prior_bounds(self, lo_bound, hi_bound):
        self.lo_inc_prior_bound = lo_bound
        self.hi_inc_prior_bound = hi_bound
    
    def set_period_prior_bounds(self, lo_bound, hi_bound):
        self.lo_period_prior_bound = lo_bound
        self.hi_period_prior_bound = hi_bound
    
    def set_ecc_prior_bounds(self, lo_bound, hi_bound):
        self.lo_ecc_prior_bound = lo_bound
        self.hi_ecc_prior_bound = hi_bound
    
    def set_dist_prior_bounds(self, lo_bound, hi_bound):
        self.lo_dist_prior_bound = lo_bound
        self.hi_dist_prior_bound = hi_bound
    
    def set_t0_prior_bounds(self, lo_bound, hi_bound):
        self.lo_t0_prior_bound = lo_bound
        self.hi_t0_prior_bound = hi_bound
    
    # Priors
    ## Using uniform priors, with radius interpolation for stellar parameters
    def lnprior(self, theta):
        # Extract model parameters from theta
        theta_index = 0
        
        Kp_ext = theta[theta_index]
        theta_index += 1
        
        if self.model_H_ext_mod:
            H_ext_mod = theta[theta_index]
            theta_index += 1
        else:
            H_ext_mod = self.default_H_ext_mod
        
        star1_rad = theta[theta_index]
        theta_index += 1
        
        star2_rad = theta[theta_index]
        theta_index += 1
        
        binary_inc = theta[theta_index]
        theta_index += 1
        
        binary_period = theta[theta_index]
        theta_index += 1
        
        if self.model_eccentricity:
            binary_ecc = theta[theta_index]
            theta_index += 1
        else:
            binary_ecc = self.default_ecc
        
        if self.model_distance:
            binary_dist = theta[theta_index]
            theta_index += 1
        else:
            binary_dist = self.default_dist
        
        t0 = theta[theta_index]
        
        ## Extinction checks
        Kp_ext_check = (self.lo_Kp_ext_prior_bound <= Kp_ext <= self.hi_Kp_ext_prior_bound)
        H_ext_mod_check = (self.lo_H_ext_mod_prior_bound <= H_ext_mod <= self.hi_H_ext_mod_prior_bound)
    
        ## Binary system configuration checks
        inc_check = (self.lo_inc_prior_bound <= binary_inc <= self.hi_inc_prior_bound)
        period_check = (self.lo_period_prior_bound <= binary_period <= self.hi_period_prior_bound)
        ecc_check = (self.lo_ecc_prior_bound <= binary_ecc <= self.hi_ecc_prior_bound)
        dist_check = (self.lo_dist_prior_bound <= binary_dist <= self.hi_dist_prior_bound)
        t0_check = (self.lo_t0_prior_bound <= t0 <= self.hi_t0_prior_bound)
        
        ## Stellar parameters check
        iso_rad_min = self.isochrone.iso_rad_min
        iso_rad_max = self.isochrone.iso_rad_max
    
        rad_check = ((star1_rad >= star2_rad) and
                     (iso_rad_min <= star1_rad <= iso_rad_max) and
                     (iso_rad_min <= star2_rad <= iso_rad_max))
        
        ## Final check and return prior
        if ((Kp_ext_check and H_ext_mod_check) and
            inc_check and period_check
            and ecc_check and dist_check and t0_check
            and rad_check):
            return 0.0
        return -np.inf
    
    # Calculate model light curve
    def calculate_model_lc(self, theta):
        # Extract model parameters from theta
        theta_index = 0
        
        Kp_ext_t = theta[theta_index]
        theta_index += 1
        
        if self.model_H_ext_mod:
            H_ext_mod_t = theta[theta_index]
            theta_index += 1
        else:
            H_ext_mod_t = self.default_H_ext_mod
        
        star1_rad_t = theta[theta_index]
        theta_index += 1
        
        star2_rad_t = theta[theta_index]
        theta_index += 1
        
        binary_inc_t = theta[theta_index]
        theta_index += 1
        
        binary_period_t = theta[theta_index]
        theta_index += 1
        
        if self.model_eccentricity:
            binary_ecc_t = theta[theta_index]
            theta_index += 1
        else:
            binary_ecc_t = self.default_ecc
        
        if self.model_distance:
            binary_dist_t = theta[theta_index]
            theta_index += 1
        else:
            binary_dist_t = self.default_dist
        
        t0_t = theta[theta_index]
        
        err_out = (np.array([-1.]), np.array([-1.]))
        
        # Add units to input parameters if necessary
        binary_inc = binary_inc_t * u.deg
        binary_period = binary_period_t * u.d
        binary_ecc = binary_ecc_t
        t0 = t0_t
        
        ## Construct tuple with binary parameters
        binary_params = (binary_period, binary_ecc, binary_inc, t0)
        
        # Calculate extinction adjustments
        Kp_ext_adj = (Kp_ext_t - self.Kp_ext)
        H_ext_adj = (((Kp_ext_t * (self.lambda_Kp / self.lambda_H)**self.ext_alpha)
                      - self.H_ext) + H_ext_mod_t)
        
        # Calculate distance modulus adjustments
        dist_mod_mag_adj = 5. * np.log10(binary_dist_t / ((self.dist).to(u.pc)).value)
        
        # Perform interpolation
        (star1_params_all, star1_params_lcfit) = self.isochrone.rad_interp(star1_rad_t)
        (star2_params_all, star2_params_lcfit) = self.isochrone.rad_interp(star2_rad_t)
        
        (star1_mass_init, star1_mass, star1_rad, star1_lum,
            star1_teff, star1_mag_Kp, star1_mag_H) = star1_params_all
        (star2_mass_init, star2_mass, star2_rad, star2_lum,
            star2_teff, star2_mag_Kp, star2_mag_H) = star2_params_all
        
        # Run single star model for reference flux calculations
        (star1_sing_mag_Kp, star1_sing_mag_H) = lc_calc.single_star_lc(
                                                    star1_params_lcfit,
                                                    use_blackbody_atm=self.use_blackbody_atm,
                                                    num_triangles=self.model_numTriangles)
        
        if (star1_sing_mag_Kp[0] == -1.) or (star1_sing_mag_H[0] == -1.):
            return err_out
        
        (star2_sing_mag_Kp, star2_sing_mag_H) = lc_calc.single_star_lc(
                                                    star2_params_lcfit,
                                                    use_blackbody_atm=self.use_blackbody_atm,
                                                    num_triangles=self.model_numTriangles)
        
        if (star2_sing_mag_Kp[0] == -1.) or (star2_sing_mag_H[0] == -1.):
            return err_out
        
        ## Apply isoc. distance modulus and isoc. extinction to single star magnitudes
        (star1_sing_mag_Kp, star1_sing_mag_H) = lc_calc.dist_ext_mag_calc(
                                                    (star1_sing_mag_Kp,
                                                    star1_sing_mag_H),
                                                    self.dist,
                                                    self.Kp_ext, self.H_ext)
        
        (star2_sing_mag_Kp, star2_sing_mag_H) = lc_calc.dist_ext_mag_calc(
                                                    (star2_sing_mag_Kp,
                                                    star2_sing_mag_H),
                                                    self.dist,
                                                    self.Kp_ext, self.H_ext)
        
        
        # Run binary star model to get binary mags
        (binary_mags_Kp, binary_mags_H) = lc_calc.binary_star_lc(
                                              star1_params_lcfit,
                                              star2_params_lcfit,
                                              binary_params,
                                              self.observation_times,
                                              use_blackbody_atm=self.use_blackbody_atm,
                                              num_triangles=self.model_numTriangles)
        if (binary_mags_Kp[0] == -1.) or (binary_mags_H[0] == -1.):
            return err_out
        
        ## Apply isoc. distance modulus and isoc. extinction to binary magnitudes
        (binary_mags_Kp, binary_mags_H) = lc_calc.dist_ext_mag_calc(
                                              (binary_mags_Kp, binary_mags_H),
                                              self.dist,
                                              self.Kp_ext, self.H_ext)
        
        # Apply flux correction to the binary magnitudes
        (binary_mags_Kp, binary_mags_H) = lc_calc.flux_adj(
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
    
    # Log Likelihood function
    def lnlike(self, theta):    
        # Extract model parameters from theta
        theta_index = 0
        
        Kp_ext_t = theta[theta_index]
        theta_index += 1
        
        if self.model_H_ext_mod:
            H_ext_mod_t = theta[theta_index]
            theta_index += 1
        else:
            H_ext_mod_t = self.default_H_ext_mod
        
        star1_rad_t = theta[theta_index]
        theta_index += 1
        
        star2_rad_t = theta[theta_index]
        theta_index += 1
        
        binary_inc_t = theta[theta_index]
        theta_index += 1
        
        binary_period_t = theta[theta_index]
        theta_index += 1
        
        if self.model_eccentricity:
            binary_ecc_t = theta[theta_index]
            theta_index += 1
        else:
            binary_ecc_t = self.default_ecc
        
        if self.model_distance:
            binary_dist_t = theta[theta_index]
            theta_index += 1
        else:
            binary_dist_t = self.default_dist
        
        t0_t = theta[theta_index]
        
        (binary_model_mags_Kp, binary_model_mags_H) = self.calculate_model_lc(theta)
        if (binary_model_mags_Kp[0] == -1.) or (binary_model_mags_H[0] == -1.):
            return -np.inf
        
        # Phase the observation times
        (kp_phase_out, h_phase_out) = lc_calc.phased_obs(
                                          self.observation_times,
                                          binary_period_t * u.d, t0_t)
        
        (kp_phased_days, kp_phases_sorted_inds, kp_model_times) = kp_phase_out
        (h_phased_days, h_phases_sorted_inds, h_model_times) = h_phase_out
        
        # Calculate log likelihood and return
        log_likelihood = np.sum((self.kp_obs_mags[kp_phases_sorted_inds] -
                             binary_model_mags_Kp)**2. /
                             (self.kp_obs_mag_errors[kp_phases_sorted_inds])**2.)
        log_likelihood += np.sum((self.h_obs_mags[h_phases_sorted_inds] -
                              binary_model_mags_H)**2. /
                              (self.h_obs_mag_errors[h_phases_sorted_inds])**2.)
    
        log_likelihood = -0.5 * log_likelihood
    
        return log_likelihood
    
    # Posterior Probability Function
    def lnprob(self, theta):
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(theta)
    
