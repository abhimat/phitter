#!/usr/bin/env python

# MCMC classes,
# for fitting with PHOEBE
# ---
# Abhimat Gautam

import phoebe
from phoebe import u
from phoebe import c as const

import numpy as np

from spisea import synthetic

from phoebe_phitter import lc_calc, blackbody_params

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.ticker import MultipleLocator

# Reference fluxes, calculated with PopStar
## Vega magnitudes (m_Vega = 0.03)
ks_filt_info = synthetic.get_filter_info('naco,Ks')

flux_ref_Ks = ks_filt_info.flux0 * (u.erg / u.s) / (u.cm**2.)

# Stellar Parameters
# stellar_params = (mass, rad, teff, mag_Kp, mag_H)

class mcmc_fitter_bb(object):
    # Filter properties
    lambda_Ks = 2.18e-6 * u.m
    dlambda_Ks = 0.35e-6 * u.m
        
    filts_lambda = {'nirc2,Kp': 2.124e-6 * u.m,
                    'nirc2,H': 1.633e-6 * u.m}
    filts_dlambda = {'nirc2,Kp': 0.351e-6 * u.m,
                     'nirc2,H': 0.296e-6 * u.m}
    
    ks_filt_info = synthetic.get_filter_info('naco,Ks')
    
    flux_ref_Ks = ks_filt_info.flux0 * (u.erg / u.s) / (u.cm**2.)
    
    # Extinction law (using Nogueras-Lara+ 2018)
    ext_alpha = 2.30
    ext_alpha_unc = 0.08
    
    # Number of triangles and atmosphere to use for binary model
    use_blackbody_atm = True
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
    
    # Model star 1 parameters
    model_star1_mass = True
    default_star1_mass = 10.0 * u.solMass
    
    model_star1_rad = True
    default_star1_rad = 10.0 * u.solRad
    
    model_star1_teff = True
    default_star1_teff = 8000. * u.K
    
    # Model star 2 parameters
    model_star2_mass = True
    default_star2_mass = 10.0 * u.solMass
    
    model_star2_rad = True
    default_star2_rad = 10.0 * u.solRad
    
    model_star2_teff = True
    default_star2_teff = 8000. * u.K
    
    # Default prior bounds
    # Extinction prior bounds
    lo_Kp_ext_prior_bound = 2.0
    hi_Kp_ext_prior_bound = 4.0
    
    lo_H_ext_mod_prior_bound = -2.0
    hi_H_ext_mod_prior_bound = 2.0
    
    H_ext_mod_alpha_sig_bound = -1.0
    
    # Star 1 prior bounds
    lo_star1_mass_prior_bound = 0.1
    hi_star1_mass_prior_bound = 20
    
    lo_star1_rad_prior_bound = 0.1
    hi_star1_rad_prior_bound = 100
    
    lo_star1_teff_prior_bound = 5000
    hi_star1_teff_prior_bound = 50000
    
    # Star 2 prior bounds
    lo_star2_mass_prior_bound = 0.1
    hi_star2_mass_prior_bound = 20
    
    lo_star2_rad_prior_bound = 0.1
    hi_star2_rad_prior_bound = 100
    
    lo_star2_teff_prior_bound = 5000
    hi_star2_teff_prior_bound = 50000
    
    # Binary system prior bounds
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
    
    # Functions to make blackbody parameters object
    def make_bb_params(self, Ks_ext, dist, filts_list=['nirc2,Kp', 'nirc2,H']):
        self.Ks_ext = Ks_ext
        
        self.dist = dist*u.pc
        self.default_dist = dist
        ## Revise prior bounds for distance
        self.lo_dist_prior_bound = 0.8 * dist
        self.hi_dist_prior_bound = 1.2 * dist
        
        # Filter info and convert extinction to fit filters
        self.filts_list = filts_list
        self.num_filts = len(self.filts_list)
        
        self.filts_info = []
        self.filts_flux_ref = np.empty(self.num_filts) *\
                                  (u.erg / u.s) / (u.cm**2.)
        
        self.filts_ext = {}
        
        for cur_filt_index in range(self.num_filts):
            cur_filt = self.filts_list[cur_filt_index]
            
            cur_filt_info = synthetic.get_filter_info(cur_filt)
            self.filts_info.append(cur_filt_info)
            
            cur_filt_flux_ref = cur_filt_info.flux0 * (u.erg / u.s) / (u.cm**2.)
            self.filts_flux_ref[cur_filt_index] = cur_filt_flux_ref
            
            # Convert from specified extinction in Ks to current filter
            cur_filt_ext = (Ks_ext *
                            (self.lambda_Ks /
                             self.filts_lambda[cur_filt])**self.ext_alpha)
            
            self.filts_ext[cur_filt] = cur_filt_ext
        
        # Make blackbody stellar params object
        self.bb_params_obj = blackbody_params.bb_stellar_params(
                                 ext=self.Ks_ext,
                                 dist=self.dist.to(u.pc).value,
                                 filts_list=self.filts_list)
        
    
    # Function to set observation filters
    def set_observation_filts(self, obs_filts):
        self.obs_filts = obs_filts
        
        self.search_filt_kp = np.where(self.obs_filts == 'kp')
        self.search_filt_h = np.where(self.obs_filts == 'h')
    
    # Function to set observation times
    def set_observation_times(self, obs_times):
        self.obs_times = obs_times
        
        self.observation_times = (obs_times[self.search_filt_kp],
                                  obs_times[self.search_filt_h])
    
    # Function to set observation mags
    def set_observation_mags(self, obs_mags, obs_mag_errors):
        self.obs_mags = obs_mags
        self.obs_mag_errors = obs_mag_errors
        
        self.kp_obs_mags = obs_mags[self.search_filt_kp]
        self.kp_obs_mag_errors = obs_mag_errors[self.search_filt_kp]
        
        self.h_obs_mags = obs_mags[self.search_filt_h]
        self.h_obs_mag_errors = obs_mag_errors[self.search_filt_h]
    
    # Function to set model mesh number of triangles
    def set_model_numTriangles(self, model_numTriangles):
        self.model_numTriangles = model_numTriangles
    
    # Function to set if using blackbody atmosphere
    def set_model_use_blackbody_atm(self, use_blackbody_atm):
        self.use_blackbody_atm = use_blackbody_atm
            
    # Functions to define prior bounds
    # Extinction priors
    def set_Kp_ext_prior_bounds(self, lo_bound, hi_bound):
        self.lo_Kp_ext_prior_bound = lo_bound
        self.hi_Kp_ext_prior_bound = hi_bound
    
    def set_H_ext_mod_prior_bounds(self, lo_bound, hi_bound):
        self.lo_H_ext_mod_prior_bound = lo_bound
        self.hi_H_ext_mod_prior_bound = hi_bound
    
    def set_H_ext_mod_extLaw_sig_prior_bounds(self, sigma_bound):
        self.H_ext_mod_alpha_sig_bound = sigma_bound
    
    # Stellar parameter priors
    def set_star1_mass_prior_bounds(self, lo_bound, hi_bound):
        self.lo_star1_mass_prior_bound = lo_bound
        self.hi_star1_mass_prior_bound = hi_bound
    
    def set_star1_rad_prior_bounds(self, lo_bound, hi_bound):
        self.lo_star1_rad_prior_bound = lo_bound
        self.hi_star1_rad_prior_bound = hi_bound
    
    def set_star1_teff_prior_bounds(self, lo_bound, hi_bound):
        self.lo_star1_teff_prior_bound = lo_bound
        self.hi_star1_teff_prior_bound = hi_bound
    
    def set_star2_mass_prior_bounds(self, lo_bound, hi_bound):
        self.lo_star2_mass_prior_bound = lo_bound
        self.hi_star2_mass_prior_bound = hi_bound
    
    def set_star2_rad_prior_bounds(self, lo_bound, hi_bound):
        self.lo_star2_rad_prior_bound = lo_bound
        self.hi_star2_rad_prior_bound = hi_bound
    
    def set_star2_teff_prior_bounds(self, lo_bound, hi_bound):
        self.lo_star2_teff_prior_bound = lo_bound
        self.hi_star2_teff_prior_bound = hi_bound
    
    # Binary system parameter priors
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
    def lnprior(self, theta):
        # Extract model parameters from theta
        theta_index = 0
        
        # Extinction model parameters
        Kp_ext = theta[theta_index]
        theta_index += 1
        
        if self.model_H_ext_mod:
            H_ext_mod = theta[theta_index]
            theta_index += 1
        else:
            H_ext_mod = self.default_H_ext_mod
        
        # Star 1 model parameters
        if self.model_star1_mass:
            star1_mass = theta[theta_index]
            theta_index += 1
        else:
            star1_mass = self.default_star1_mass
        
        if self.model_star1_rad:
            star1_rad = theta[theta_index]
            theta_index += 1
        else:
            star1_rad = self.default_star1_rad
        
        if self.model_star1_teff:
            star1_teff = theta[theta_index]
            theta_index += 1
        else:
            star1_teff = self.default_star1_teff
        
        # Star 2 model parameters
        if self.model_star2_mass:
            star2_mass = theta[theta_index]
            theta_index += 1
        else:
            star2_mass = self.default_star2_mass
        
        if self.model_star2_rad:
            star2_rad = theta[theta_index]
            theta_index += 1
        else:
            star2_rad = self.default_star2_rad
        
        if self.model_star2_teff:
            star2_teff = theta[theta_index]
            theta_index += 1
        else:
            star2_teff = self.default_star2_teff
        
        # Binary model parameters
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
        Kp_ext_check = (self.lo_Kp_ext_prior_bound <= Kp_ext <=
                        self.hi_Kp_ext_prior_bound)
        
        H_ext_mod_check = True
        
        H_ext_mod_bound_oneSig = 1.0
        if self.H_ext_mod_alpha_sig_bound == -1.0:
            H_ext_mod_check = (self.lo_H_ext_mod_prior_bound <= H_ext_mod <=
                               self.hi_H_ext_mod_prior_bound)
        else:
            ### H extinction expected by Kp extinction
            H_ext = Kp_ext * ((self.filts_lambda['nirc2,Kp']/self.filts_lambda['nirc2,H'])**(self.ext_alpha))
            
            ### Bounds given by current extinction and uncertainty on extinction law
            H_ext_mod_bound_hi = Kp_ext * ((self.filts_lambda['nirc2,Kp']/self.filts_lambda['nirc2,H'])**(self.ext_alpha + self.ext_alpha_unc))
            H_ext_mod_bound_lo = Kp_ext * ((self.filts_lambda['nirc2,Kp']/self.filts_lambda['nirc2,H'])**(self.ext_alpha - self.ext_alpha_unc))
            
            ### Subtract off the H extinction expected by the Kp extinction to get mod
            H_ext_mod_bound_hi = H_ext_mod_bound_hi - H_ext
            H_ext_mod_bound_lo = H_ext - H_ext_mod_bound_lo
            
            H_ext_mod_bound_oneSig = np.max(np.abs([H_ext_mod_bound_hi, H_ext_mod_bound_lo]))
        
        # Stellar parameters checks
        star1_mass_check = True
        if self.model_star1_mass:
            star1_mass_check = (self.lo_star1_mass_prior_bound <= star1_mass <=
                                self.hi_star1_mass_prior_bound)
        
        star1_rad_check = True
        if self.model_star1_rad:
            star1_rad_check = (self.lo_star1_rad_prior_bound <= star1_rad <=
                               self.hi_star1_rad_prior_bound)
        
        star1_teff_check = True
        if self.model_star1_teff:
            star1_teff_check = (self.lo_star1_teff_prior_bound <= star1_teff <=
                                self.hi_star1_teff_prior_bound)
        
        star1_checks = star1_mass_check and star1_rad_check and star1_teff_check
        
        
        star2_mass_check = True
        if self.model_star2_mass:
            star2_mass_check = (self.lo_star2_mass_prior_bound <= star2_mass <=
                                self.hi_star2_mass_prior_bound)
        
        star2_rad_check = True
        if self.model_star2_rad:
            star2_rad_check = (self.lo_star2_rad_prior_bound <= star2_rad <=
                               self.hi_star2_rad_prior_bound)
        
        star2_teff_check = True
        if self.model_star2_teff:
            star2_teff_check = (self.lo_star2_teff_prior_bound <= star2_teff <=
                                self.hi_star2_teff_prior_bound)
        
        star2_checks = star2_mass_check and star2_rad_check and star2_teff_check
                
        ## Binary system configuration checks
        inc_check = (self.lo_inc_prior_bound <= binary_inc <=
                     self.hi_inc_prior_bound)
        period_check = (self.lo_period_prior_bound <= binary_period <=
                        self.hi_period_prior_bound)
        ecc_check = (self.lo_ecc_prior_bound <= binary_ecc <=
                     self.hi_ecc_prior_bound)
        dist_check = (self.lo_dist_prior_bound <= binary_dist <= self.hi_dist_prior_bound)
        t0_check = (self.lo_t0_prior_bound <= t0 <= self.hi_t0_prior_bound)
        
        # print(Kp_ext_check)
        # print(H_ext_mod_check)
        # print(inc_check)
        # print(period_check)
        # print(ecc_check)
        # print(dist_check)
        # print(t0_check)
        # print(star1_checks)
        # print(star2_checks)
        
        ## Final check and return prior
        if self.H_ext_mod_alpha_sig_bound == -1.0:  # If doing simple H_ext check
            if ((Kp_ext_check and H_ext_mod_check) and
                inc_check and period_check
                and ecc_check and dist_check and t0_check
                and star1_checks and star2_checks):
                return 0.0
        else:   # Else doing Gaussian prior check on H_ext
            if (Kp_ext_check
                and inc_check and period_check
                and ecc_check and dist_check and t0_check
                and star1_checks and star2_checks):
                
                # Return gaussian prior for H_ext_mod parameter
                log_prior = np.log(1.0/(np.sqrt(2*np.pi)*H_ext_mod_bound_oneSig))
                log_prior = (log_prior - 
                             0.5 * (H_ext_mod**2) / (H_ext_mod_bound_oneSig**2))
                return log_prior
        
        # If here at this point, all previous checks failed
        return -np.inf
    
    # Calculate model light curve
    def calculate_model_lc(self, theta):
        # Extract model parameters from theta
        theta_index = 0
        
        # Extinction model parameters
        Kp_ext = theta[theta_index]
        theta_index += 1
        
        if self.model_H_ext_mod:
            H_ext_mod = theta[theta_index]
            theta_index += 1
        else:
            H_ext_mod = self.default_H_ext_mod
        
        # Star 1 model parameters
        if self.model_star1_mass:
            star1_mass = theta[theta_index] * u.solMass
            theta_index += 1
        else:
            star1_mass = self.default_star1_mass
        
        if self.model_star1_rad:
            star1_rad = theta[theta_index] * u.solRad
            theta_index += 1
        else:
            star1_rad = self.default_star1_rad
        
        if self.model_star1_teff:
            star1_teff = theta[theta_index] * u.K
            theta_index += 1
        else:
            star1_teff = self.default_star1_teff
        
        # Star 2 model parameters
        if self.model_star2_mass:
            star2_mass = theta[theta_index] * u.solMass
            theta_index += 1
        else:
            star2_mass = self.default_star2_mass
        
        if self.model_star2_rad:
            star2_rad = theta[theta_index] * u.solRad
            theta_index += 1
        else:
            star2_rad = self.default_star2_rad
        
        if self.model_star2_teff:
            star2_teff = theta[theta_index] * u.K
            theta_index += 1
        else:
            star2_teff = self.default_star2_teff
        
        # Binary model parameters
        binary_inc = theta[theta_index] * u.deg
        theta_index += 1
        
        binary_period = theta[theta_index] * u.d
        theta_index += 1
        
        if self.model_eccentricity:
            binary_ecc = theta[theta_index]
            theta_index += 1
        else:
            binary_ecc = self.default_ecc
        
        if self.model_distance:
            binary_dist = theta[theta_index] * u.pc
            theta_index += 1
        else:
            binary_dist = self.default_dist
        
        t0 = theta[theta_index]
        
        err_out = (np.array([-1.]), np.array([-1.]))
        
        ## Construct tuple with binary parameters
        binary_params = (binary_period, binary_ecc, binary_inc, t0)
        
        # Calculate extinction adjustments
        filt_ext_adj = np.empty(self.num_filts)
        
        Kp_ext_adj = (Kp_ext - self.filts_ext['nirc2,Kp'])
        H_ext_adj = (((Kp_ext * (self.filts_lambda['nirc2,Kp'] / self.filts_lambda['nirc2,H'])**self.ext_alpha)
                      - self.filts_ext['nirc2,H']) + H_ext_mod)
        
        filt_ext_adj = np.array([Kp_ext_adj, H_ext_adj])
        
        # Calculate distance modulus adjustments
        dist_mod_mag_adj = 5. * np.log10(binary_dist / ((self.dist).to(u.pc)).value)
        
        # Perform interpolation
        (star1_params_all,
         star1_params_lcfit) = self.bb_params_obj.calc_stellar_params(
                                 star1_mass, star1_rad, star1_teff)
        (star2_params_all,
         star2_params_lcfit) = self.bb_params_obj.calc_stellar_params(
                                 star2_mass, star2_rad, star2_teff)
        
        (star1_mass_init, star1_mass, star1_rad, star1_lum, star1_teff, star1_logg,
            [star1_mag_Kp, star1_mag_H],
            [star1_pblum_Kp, star1_pblum_H]) = star1_params_all
        (star2_mass_init, star2_mass, star2_rad, star2_lum, star2_teff, star2_logg,
            [star2_mag_Kp, star2_mag_H],
            [star2_pblum_Kp, star2_pblum_H]) = star2_params_all
        
        # Run binary star model to get binary mags
        (binary_mags_Kp, binary_mags_H) = lc_calc.binary_star_lc(
                                              star1_params_lcfit,
                                              star2_params_lcfit,
                                              binary_params,
                                              self.observation_times,
                                              use_blackbody_atm=self.use_blackbody_atm,
                                              use_compact_object=True,
                                              num_triangles=self.model_numTriangles)
        if (binary_mags_Kp[0] == -1.) or (binary_mags_H[0] == -1.):
            return err_out
        
        ## Apply isoc. distance modulus and isoc. extinction to binary magnitudes
        (binary_mags_Kp, binary_mags_H) = lc_calc.dist_ext_mag_calc(
                                              (binary_mags_Kp, binary_mags_H),
                                              self.dist,
                                              self.filts_ext['nirc2,Kp'],
                                              self.filts_ext['nirc2,H'])
        
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
        
        # Extinction model parameters
        Kp_ext = theta[theta_index]
        theta_index += 1
        
        if self.model_H_ext_mod:
            H_ext_mod = theta[theta_index]
            theta_index += 1
        else:
            H_ext_mod = self.default_H_ext_mod
        
        # Star 1 model parameters
        if self.model_star1_mass:
            star1_mass = theta[theta_index] * u.solMass
            theta_index += 1
        else:
            star1_mass = self.default_star1_mass
        
        if self.model_star1_rad:
            star1_rad = theta[theta_index] * u.solRad
            theta_index += 1
        else:
            star1_rad = self.default_star1_rad
        
        if self.model_star1_teff:
            star1_teff = theta[theta_index] * u.K
            theta_index += 1
        else:
            star1_teff = self.default_star1_teff
        
        # Star 2 model parameters
        if self.model_star2_mass:
            star2_mass = theta[theta_index] * u.solMass
            theta_index += 1
        else:
            star2_mass = self.default_star2_mass
        
        if self.model_star2_rad:
            star2_rad = theta[theta_index] * u.solRad
            theta_index += 1
        else:
            star2_rad = self.default_star2_rad
        
        if self.model_star2_teff:
            star2_teff = theta[theta_index] * u.K
            theta_index += 1
        else:
            star2_teff = self.default_star2_teff
        
        # Binary model parameters
        binary_inc = theta[theta_index] * u.deg
        theta_index += 1
        
        binary_period = theta[theta_index] * u.d
        theta_index += 1
        
        if self.model_eccentricity:
            binary_ecc = theta[theta_index]
            theta_index += 1
        else:
            binary_ecc = self.default_ecc
        
        if self.model_distance:
            binary_dist = theta[theta_index] * u.pc
            theta_index += 1
        else:
            binary_dist = self.default_dist
        
        t0 = theta[theta_index]
        
        # Calculate light curve
        (binary_model_mags_Kp, binary_model_mags_H) = self.calculate_model_lc(theta)
        
        if (binary_model_mags_Kp[0] == -1.) or (binary_model_mags_H[0] == -1.):
            return -np.inf
        
        # Phase the observation times
        (kp_phase_out, h_phase_out) = lc_calc.phased_obs(
                                          self.observation_times,
                                          binary_period, t0)
        
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
