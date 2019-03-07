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
    
    def __init__(self):
        return
    
    # Function to make and store isochrone
    def make_isochrone(self, age, Ks_ext, dist, phase):
        self.Ks_ext = Ks_ext
        self.dist = dist*u.pc
        self.age = age
        
        self.isochrone = isoc_interp.isochrone_mist(age=age,
                             ext=Ks_ext, dist=dist, phase=phase)
        
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
    
    # Priors
    ## Using uniform priors, with radius interpolation for stellar parameters
    def lnprior(self, theta):
        (Kp_ext, H_ext_mod,
         star1_rad, star2_rad,
         binary_inc, binary_period,
         binary_ecc, t0) = theta
        
        ## Extinction checks
        Kp_ext_check = (2.0 <= Kp_ext <= 3.6)
        H_ext_mod_check = (-1.0 <= H_ext_mod <= 1.0)
    
        ## Binary system configuration checks
        inc_check = (0. <= binary_inc <= 180.)
        period_check = (77.5 <= binary_period <= 80.5)
        ecc_check = (-0.1 <= binary_ecc <= 0.1)
        t0_check = (53720.0 <= t0 <= 53840.0)
    
        ## Stellar parameters check
        iso_rad_min = self.isochrone.iso_rad_min
        iso_rad_max = self.isochrone.iso_rad_max
    
        rad_check = ((iso_rad_min <= star1_rad <= iso_rad_max) and
                     (iso_rad_min <= star2_rad <= iso_rad_max))
    
        ## Final check and return prior
        if ((Kp_ext_check and H_ext_mod_check) and
            inc_check and period_check
            and ecc_check and t0_check
            and rad_check):
            return 0.0
        return -np.inf

    # Log Likelihood function
    def lnlike(self, theta):
        (Kp_ext_t, H_ext_mod_t,
         star1_rad_t, star2_rad_t,
         binary_inc_t, binary_period_t,
         binary_ecc_t, t0_t) = theta
        
        # Add units to input parameters if necessary
        binary_inc = binary_inc_t * u.deg
        binary_period = binary_period_t * u.d
        binary_ecc = binary_ecc_t
        t0 = t0_t
        
        ## Construct tuple with binary parameters
        binary_params = (binary_period, binary_ecc, binary_inc, t0)
        
        ### Phase the observations
        (kp_phase_out, h_phase_out) = lc_calc.phased_obs(
                                          self.observation_times,
                                          binary_period, t0)
        
        (kp_phased_days, kp_phases_sorted_inds, kp_model_times) = kp_phase_out
        (h_phased_days, h_phases_sorted_inds, h_model_times) = h_phase_out
        
        # Calculate extinction adjustments
        Kp_ext_adj = (Kp_ext_t - self.Kp_ext)
        H_ext_adj = (((Kp_ext_t * (self.lambda_Kp / self.lambda_H)**self.ext_alpha)
                      - self.H_ext) + H_ext_mod_t)
        
        
        # Perform interpolation
        (star1_params_all, star1_params_lcfit) = self.isochrone.rad_interp(star1_rad_t)
        (star2_params_all, star2_params_lcfit) = self.isochrone.rad_interp(star2_rad_t)
        
        (star1_mass_init, star1_mass, star1_rad, star1_lum,
            star1_teff, star1_mag_Kp, star1_mag_H) = star1_params_all
        (star2_mass_init, star2_mass, star2_rad, star2_lum,
            star2_teff, star2_mag_Kp, star2_mag_H) = star2_params_all
        
        # Run single star model for reference flux calculations
        (star1_sing_mag_Kp, star1_sing_mag_H) = lc_calc.single_star_lc(star1_params_lcfit)
        
        if (star1_sing_mag_Kp[0] == -1.) or (star1_sing_mag_H[0] == -1.):
            return -np.inf
        
        (star2_sing_mag_Kp, star2_sing_mag_H) = lc_calc.single_star_lc(star2_params_lcfit)
        
        if (star2_sing_mag_Kp[0] == -1.) or (star2_sing_mag_H[0] == -1.):
            return -np.inf
        
        ## Apply distance modulus and isoc. extinction to single star magnitudes
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
                                              self.observation_times)
        if (binary_mags_Kp[0] == -1.) or (binary_mags_H[0] == -1.):
            return -np.inf
        
        ## Apply distance modulus and isoc. extinction to binary magnitudes
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
        
        # Calculate log likelihood and return
        log_likelihood = np.sum((self.kp_obs_mags[kp_phases_sorted_inds] -
                             binary_mags_Kp)**2. /
                             (self.kp_obs_mag_errors[kp_phases_sorted_inds])**2.)
        log_likelihood += np.sum((self.h_obs_mags[h_phases_sorted_inds] -
                              binary_mags_H)**2. /
                              (self.h_obs_mag_errors[h_phases_sorted_inds])**2.)
    
        log_likelihood = -0.5 * log_likelihood
    
        return log_likelihood
    
    # Posterior Probability Function
    def lnprob(self, theta):
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(theta)
    
