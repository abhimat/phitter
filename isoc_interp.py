#!/usr/bin/env python

# Isochrone interpolation functions,
# using isochrones generated with PopStar
# ---
# Abhimat Gautam

from spisea import synthetic, evolution, atmospheres, reddening
from spisea.imf import imf, multiplicity

from phoebe import u
from phoebe import c as const

import numpy as np

# Dictionary to help map phases to corresponding code in the MIST isochrone
mist_phase_dict = {}
mist_phase_dict['PMS'] = -1
mist_phase_dict['MS'] = 0
mist_phase_dict['RGB'] = 2
mist_phase_dict['CHeB'] = 3
mist_phase_dict['EAGB'] = 4
mist_phase_dict['TPAGB'] = 5
mist_phase_dict['postAGB'] = 6
mist_phase_dict['WR'] = 9

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

v_filt_info = synthetic.get_filter_info('ubv,V')

flux_ref_Ks = ks_filt_info.flux0 * (u.erg / u.s) / (u.cm**2.)
flux_ref_Kp = kp_filt_info.flux0 * (u.erg / u.s) / (u.cm**2.)
flux_ref_H = h_filt_info.flux0 * (u.erg / u.s) / (u.cm**2.)

flux_ref_V = v_filt_info.flux0 * (u.erg / u.s) / (u.cm**2.)


class isochrone_mist(object):
    filts_list = ['nirc2,Kp', 'nirc2,H']
    
    def __init__(self, age=3.9e6, ext=2.63, dist=7.971e3, met=0.0, phase=None,
                 use_atm_func='merged',
                 filts_list=['nirc2,Kp', 'nirc2,H']):
        log_age = np.log10(age)
        
        self.log_age = log_age
        self.A_Ks = ext
        self.dist = dist
        self.met = met
        
        # Specify filters and get filter information
        self.filts_list = filts_list
        self.num_filts = len(self.filts_list)
        
        self.filts_info = []
        self.filts_flux_ref = np.empty(self.num_filts) *\
                                  (u.erg / u.s) / (u.cm**2.)
        for cur_filt_index in range(self.num_filts):
            cur_filt = self.filts_list[cur_filt_index]
            
            cur_filt_info = synthetic.get_filter_info(cur_filt)
            self.filts_info.append(cur_filt_info)
            
            cur_filt_flux_ref = cur_filt_info.flux0 * (u.erg / u.s) / (u.cm**2.)
            self.filts_flux_ref[cur_filt_index] = cur_filt_flux_ref
        
        # Evolution/Atmosphere Models
        evo_model = evolution.MISTv1()
        
        if use_atm_func == 'merged':
            atm_func = atmospheres.get_merged_atmosphere
        elif use_atm_func == 'castelli':
            atm_fun = atmospheres.get_castelli_atmosphere
        elif use_atm_func == 'phoenix':
            atm_func = atmospheres.get_phoenixv16_atmosphere
        
        # Extinction law    
        red_law = reddening.RedLawNoguerasLara18()
        self.ext_alpha = 2.30
    
        ## Calculate extinctions implied by isochrone extinction
        self.A_Kp = self.A_Ks * (lambda_Ks / lambda_Kp)**self.ext_alpha
        self.A_H = self.A_Ks * (lambda_Ks / lambda_H)**self.ext_alpha
        
        # Create an isochrone with the given parameters        
        self.iso_curAge = synthetic.IsochronePhot(self.log_age, self.A_Ks, self.dist,
                                                  evo_model=evo_model,
                                                  atm_func=atm_func,
                                                  red_law=red_law,
                                                  metallicity=self.met,
                                                  filters=self.filts_list)
        
        ## Create another isochrone for absolute mags / passband luminosities
        self.iso_absMag = synthetic.IsochronePhot(self.log_age, 0.0, 10.0,
                                                  evo_model=evo_model,
                                                  atm_func=atm_func,
                                                  red_law=red_law,
                                                  metallicity=self.met,
                                                  filters=self.filts_list)
        
        
        # Save out specific stellar parameter columns needed
        ## If needing specific phase, draw it out before saving
        if phase is not None:
            phase_check = np.where(self.iso_curAge.points['phase'] == mist_phase_dict[phase])
        else:
            phase_check = np.where(self.iso_curAge.points['phase'] >= -1)
        
        self.iso_mass_init = (self.iso_curAge.points['mass'][phase_check]).to(u.solMass)
        self.iso_mass = (self.iso_curAge.points['mass_current'][phase_check]).to(u.solMass)
        self.iso_rad = (self.iso_curAge.points['R'][phase_check]).to(u.solRad)
        self.iso_lum = (self.iso_curAge.points['L'][phase_check]).to(u.solLum)
        self.iso_teff = (self.iso_curAge.points['Teff'][phase_check]).to(u.K)
        self.iso_logg = self.iso_curAge.points['logg'][phase_check]
        
        self.iso_mag = {}
        for filt in self.filts_list:
            filt_dict_name = filt.replace(',', '_')
            filt_dict_name = filt_dict_name.replace('wfc3_ir', 'hst')
            
            self.iso_mag[filt] = self.iso_curAge.points['m_' + filt_dict_name][phase_check]
        
        ## Stellar parameters from the absolute magnitude isochrones
        self.iso_absMag_mass_init = (self.iso_absMag.points['mass'][phase_check]).to(u.solMass)
        self.iso_absMag_mass = (self.iso_absMag.points['mass_current'][phase_check]).to(u.solMass)
        self.iso_absMag_rad = (self.iso_absMag.points['R'][phase_check]).to(u.solRad)
        
        self.iso_absMag_mag = {}
        for filt in self.filts_list:
            filt_dict_name = filt.replace(',', '_')
            filt_dict_name = filt_dict_name.replace('wfc3_ir', 'hst')
            
            self.iso_absMag_mag[filt] = self.iso_absMag.points['m_' + filt_dict_name][phase_check]
        
        ## Maximum bounds on the radius in isochrone
        self.iso_rad_min = np.min(self.iso_rad).value
        self.iso_rad_max = np.max(self.iso_rad).value
        
        ## Maximum bounds on the initial mass in isochrone
        self.iso_mass_init_min = np.min(self.iso_mass_init).value
        self.iso_mass_init_max = np.max(self.iso_mass_init).value
    
    def rad_interp(self, star_rad_interp):
        # Reverse isochrones, if radius not increasing, for numpy interpolation to work
        if self.iso_rad[-1] < self.iso_rad[0]:
            self.iso_mass_init = self.iso_mass_init[::-1]
            self.iso_mass = self.iso_mass[::-1]
            self.iso_rad = self.iso_rad[::-1]
            self.iso_lum = self.iso_lum[::-1]
            self.iso_teff = self.iso_teff[::-1]
            self.iso_logg = self.iso_logg[::-1]

            for filt in self.filts_list:
                self.iso_mag[filt] = (self.iso_mag[filt])[::-1]
                self.iso_absMag_mag[filt] = (self.iso_absMag[filt])[::-1]
        
        star_rad = star_rad_interp * u.solRad
        
        star_mass_init = np.interp(star_rad, self.iso_rad, self.iso_mass_init)
        star_mass = np.interp(star_rad, self.iso_rad, self.iso_mass)
        star_lum = np.interp(star_rad, self.iso_rad, self.iso_lum)
        star_teff = np.interp(star_rad, self.iso_rad, self.iso_teff)
        star_logg = np.interp(star_rad, self.iso_rad, self.iso_logg)
        
        star_mags = np.empty(self.num_filts)
        star_absMags = np.empty(self.num_filts)
        
        for cur_filt_index in range(self.num_filts):
            cur_filt = self.filts_list[cur_filt_index]
            cur_filt_flux_ref = self.filts_flux_ref[cur_filt_index]
            
            star_mags[cur_filt_index] = np.interp(star_rad, self.iso_rad,
                                                  self.iso_mag[cur_filt])
            
            star_absMags[cur_filt_index] = np.interp(star_rad,
                                                     self.iso_absMag_rad,
                                                     self.iso_absMag_mag[cur_filt])
        
        # Passband luminosities
        star_pblums = self.calc_pblums(star_absMags)
        
        # Export tuple with all parameters and tuple with only parameters needed for lc fit
        stellar_params_all = (star_mass_init, star_mass,
                              star_rad, star_lum,
                              star_teff, star_logg,
                              star_mags, star_pblums)
        stellar_params_lcfit = (star_mass, star_rad,
                                star_teff, star_logg,
                                star_mags, star_pblums)
        
        return stellar_params_all, stellar_params_lcfit
    
    def mass_init_interp(self, star_mass_init_interp):
        star_mass_init = star_mass_init_interp * u.solMass
        
        star_mass = np.interp(star_mass_init, self.iso_mass_init, self.iso_mass)
        star_rad = np.interp(star_mass_init, self.iso_mass_init, self.iso_rad)
        star_lum = np.interp(star_mass_init, self.iso_mass_init, self.iso_lum)
        star_teff = np.interp(star_mass_init, self.iso_mass_init, self.iso_teff)
        star_logg = np.interp(star_mass_init, self.iso_mass_init, self.iso_logg)
        
        star_mags = np.empty(self.num_filts)
        star_absMags = np.empty(self.num_filts)
        
        for cur_filt_index in range(self.num_filts):
            cur_filt = self.filts_list[cur_filt_index]
            cur_filt_flux_ref = self.filts_flux_ref[cur_filt_index]
            
            star_mags[cur_filt_index] = np.interp(star_mass_init,
                                                  self.iso_mass_init,
                                                  self.iso_mag[cur_filt])
            
            star_absMags[cur_filt_index] = np.interp(star_mass_init,
                                                     self.iso_absMag_mass_init,
                                                     self.iso_absMag_mag[cur_filt])
        
        # Passband luminosities
        star_pblums = self.calc_pblums(star_absMags)
        
        # Export tuple with all parameters and tuple with only parameters needed for lc fit
        stellar_params_all = (star_mass_init, star_mass,
                              star_rad, star_lum,
                              star_teff, star_logg,
                              star_mags, star_pblums)
        stellar_params_lcfit = (star_mass, star_rad,
                                star_teff, star_logg,
                                star_mags, star_pblums)
        
        return stellar_params_all, stellar_params_lcfit
    
    def mass_interp(self, star_mass_interp):
        star_mass = star_mass_interp * u.solMass
        
        star_mass_init = np.interp(star_mass, self.iso_mass, self.iso_mass_init)
        star_rad = np.interp(star_mass, self.iso_mass, self.iso_rad)
        star_lum = np.interp(star_mass, self.iso_mass, self.iso_lum)
        star_teff = np.interp(star_mass, self.iso_mass, self.iso_teff)
        star_logg = np.interp(star_mass, self.iso_mass, self.iso_logg)
        
        star_mags = np.empty(self.num_filts)
        star_absMags = np.empty(self.num_filts)
        
        for cur_filt_index in range(self.num_filts):
            cur_filt = self.filts_list[cur_filt_index]
            cur_filt_flux_ref = self.filts_flux_ref[cur_filt_index]
            
            star_mags[cur_filt_index] = np.interp(star_mass,
                                                  self.iso_mass,
                                                  self.iso_mag[cur_filt])
            
            star_absMags[cur_filt_index] = np.interp(star_mass,
                                                     self.iso_absMag_mass,
                                                     self.iso_absMag_mag[cur_filt])
        
        # Passband luminosities
        star_pblums = self.calc_pblums(star_absMags)
        
        # Export tuple with all parameters and tuple with only parameters needed for lc fit
        stellar_params_all = (star_mass_init, star_mass,
                              star_rad, star_lum,
                              star_teff, star_logg,
                              star_mags, star_pblums)
        stellar_params_lcfit = (star_mass, star_rad,
                                star_teff, star_logg,
                                star_mags, star_pblums)
        
        return stellar_params_all, stellar_params_lcfit
    
    
    def calc_pblums(self, filt_absMags):
        # Calculate luminosities in each filter
        filt_pblums = np.empty(self.num_filts) * u.solLum
        
        for cur_filt_index in range(self.num_filts):
            cur_filt_flux_ref = self.filts_flux_ref[cur_filt_index]
            cur_filt_absMag = filt_absMags[cur_filt_index]
            
            # Convert current filter magnitude into flux
            cur_filt_flux = (cur_filt_flux_ref *
                             (10.**((cur_filt_absMag - 0.03) / -2.5)))
            
            # Calculate passband luminosity
            cur_filt_pblum = (cur_filt_flux *
                              (4. * np.pi * (10. * u.pc)**2.))
            
            filt_pblums[cur_filt_index] = cur_filt_pblum.to(u.solLum)
        
        return filt_pblums
