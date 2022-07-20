#!/usr/bin/env python

# Isochrone interpolation functions,
# using isochrones generated with PopStar
# ---
# Abhimat Gautam

from popstar import synthetic, evolution, atmospheres, reddening
from popstar.imf import imf, multiplicity

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
# lambda_Ks = 2.18e-6 * u.m
# dlambda_Ks = 0.35e-6 * u.m

lambda_Ks = 2.159e-6 * u.m
dlambda_Ks = 0.262e-6 * u.m

lambda_H = 1.662e-6 * u.m
dlambda_H = 0.251e-6 * u.m

# Reference fluxes, calculated with PopStar
## Vega magnitudes (m_Vega = 0.03)
ks_filt_info = synthetic.get_filter_info('2mass,Ks')
h_filt_info = synthetic.get_filter_info('2mass,H')

v_filt_info = synthetic.get_filter_info('ubv,V')

flux_ref_Ks = ks_filt_info.flux0 * (u.erg / u.s) / (u.cm**2.)
flux_ref_H = h_filt_info.flux0 * (u.erg / u.s) / (u.cm**2.)

flux_ref_V = v_filt_info.flux0 * (u.erg / u.s) / (u.cm**2.)


class isochrone_mist(object):
    filt_list = ['2mass,Ks', '2mass,H']
    
    def __init__(self, age=3.9e6, ext=2.63, dist=7.971e3, met=0.0, phase=None,
                 use_atm_func='merged'):
        log_age = np.log10(age)
        
        self.log_age = log_age
        self.A_Ks = ext
        self.dist = dist
        self.met = met
        
        # Evolution/Atmosphere Models
        evo_model = evolution.MISTv1()
        
        if use_atm_func == 'merged':
            atm_func = atmospheres.get_merged_atmosphere
        elif use_atm_func == 'phoenix':
            atm_func = atmospheres.get_phoenixv16_atmosphere
        
        # Extinction law    
        red_law = reddening.RedLawNoguerasLara18()
        self.ext_alpha = 2.30
    
        ## Calculate extinctions implied by isochrone extinction
        self.A_Ks = self.A_Ks * (lambda_Ks / lambda_Ks)**self.ext_alpha
        self.A_H = self.A_Ks * (lambda_Ks / lambda_H)**self.ext_alpha
        
        # Create an isochhrone with the given parameters        
        self.iso_curAge = synthetic.IsochronePhot(self.log_age, self.A_Ks, self.dist,
                                                  evo_model=evo_model,
                                                  atm_func=atm_func,
                                                  red_law=red_law,
                                                  metallicity=self.met,
                                                  filters=self.filt_list)
        
        # Save out specific stellar parameter columns needed
        ## If needing specific phase, draw it out before saving
        if phase is not None:
            phase_check = np.where(self.iso_curAge.points['phase'] == mist_phase_dict[phase])
        else:
            phase_check = np.where(self.iso_curAge.points['phase'] >= -1)
        
        self.iso_mass_init = self.iso_curAge.points['mass'][phase_check]
        self.iso_mass = self.iso_curAge.points['mass_current'][phase_check]
        self.iso_rad = (self.iso_curAge.points['R'][phase_check]).to(u.solRad)
        self.iso_lum = self.iso_curAge.points['L'][phase_check]
        self.iso_teff = self.iso_curAge.points['Teff'][phase_check]
        
        self.iso_mag_Ks = self.iso_curAge.points['m_2mass_Ks'][phase_check]
        self.iso_mag_H = self.iso_curAge.points['m_2mass_H'][phase_check]
        
        self.iso_rad_min = np.min(self.iso_rad).value
        self.iso_rad_max = np.max(self.iso_rad).value
    
    def rad_interp(self, star_rad_interp):
        # Reverse isochrones, if radius not increasing, for numpy interpolation to work
        if self.iso_rad[-1] < self.iso_rad[0]:
            self.iso_mass_init = self.iso_mass_init[::-1]
            self.iso_mass = self.iso_mass[::-1]
            self.iso_rad = self.iso_rad[::-1]
            self.iso_lum = self.iso_lum[::-1]
            self.iso_teff = self.iso_teff[::-1]

            self.iso_mag_Ks = self.iso_mag_Ks[::-1]
            self.iso_mag_H = self.iso_mag_H[::-1]
        
        star_rad = star_rad_interp * u.solRad
        
        star_mass_init = np.interp(star_rad_interp, self.iso_rad, self.iso_mass_init) * u.solMass
        star_mass = np.interp(star_rad_interp, self.iso_rad, self.iso_mass) * u.solMass
        star_lum = np.interp(star_rad_interp, self.iso_rad, self.iso_lum) * u.W
        star_teff = np.interp(star_rad_interp, self.iso_rad, self.iso_teff) * u.K
        star_mag_Ks = np.interp(star_rad_interp, self.iso_rad, self.iso_mag_Ks)
        star_mag_H = np.interp(star_rad_interp, self.iso_rad, self.iso_mag_H)
        
        # Passband luminosities
        star_pblum_Ks, star_pblum_H = self.calc_pb_lums(star_mag_Ks, star_mag_H)
        
        # Export tuple with all parameters and tuple with only parameters needed for lc fit
        stellar_params_all = (star_mass_init, star_mass, star_rad, star_lum, star_teff,
                              star_mag_Ks, star_mag_H,
                              star_pblum_Ks, star_pblum_H)
        stellar_params_lcfit = (star_mass, star_rad, star_teff,
                                star_mag_Ks, star_mag_H,
                                star_pblum_Ks, star_pblum_H)
        
        return stellar_params_all, stellar_params_lcfit
    
    def mass_init_interp(self, star_mass_init_interp):
        star_mass_init = star_mass_init_interp * u.solMass
        
        star_mass = np.interp(star_mass_init_interp, self.iso_mass_init, self.iso_mass) * u.solMass
        star_rad = np.interp(star_mass_init_interp, self.iso_mass_init, self.iso_rad) * u.solRad
        star_lum = np.interp(star_mass_init_interp, self.iso_mass_init, self.iso_lum) * u.W
        star_teff = np.interp(star_mass_init_interp, self.iso_mass_init, self.iso_teff) * u.K
        star_mag_Ks = np.interp(star_mass_init_interp, self.iso_mass_init, self.iso_mag_Ks)
        star_mag_H = np.interp(star_mass_init_interp, self.iso_mass_init, self.iso_mag_H)
        
        # Passband luminosities
        star_pblum_Ks, star_pblum_H = self.calc_pb_lums(star_mag_Ks, star_mag_H)
        
        # Export tuple with all parameters and tuple with only parameters needed for lc fit
        stellar_params_all = (star_mass_init, star_mass, star_rad, star_lum, star_teff,
                              star_mag_Ks, star_mag_H,
                              star_pblum_Ks, star_pblum_H)
        stellar_params_lcfit = (star_mass, star_rad, star_teff,
                                star_mag_Ks, star_mag_H,
                                star_pblum_Ks, star_pblum_H)
        
        return stellar_params_all, stellar_params_lcfit
    
    def mass_interp(self, star_mass_interp):
        # star_mass_init = star_mass_init_interp * u.solMass
        star_mass = star_mass_interp * u.solMass
        
        star_mass_init = np.interp(star_mass_interp, self.iso_mass, self.iso_mass_init) * u.solMass
        star_rad = np.interp(star_mass_interp, self.iso_mass, self.iso_rad) * u.solRad
        star_lum = np.interp(star_mass_interp, self.iso_mass, self.iso_lum) * u.W
        star_teff = np.interp(star_mass_interp, self.iso_mass, self.iso_teff) * u.K
        star_mag_Ks = np.interp(star_mass_interp, self.iso_mass, self.iso_mag_Ks)
        star_mag_H = np.interp(star_mass_interp, self.iso_mass, self.iso_mag_H)
        
        # Passband luminosities
        star_pblum_Ks, star_pblum_H = self.calc_pb_lums(star_mag_Ks, star_mag_H)
        
        # Export tuple with all parameters and tuple with only parameters needed for lc fit
        stellar_params_all = (star_mass_init, star_mass, star_rad, star_lum, star_teff,
                              star_mag_Ks, star_mag_H,
                              star_pblum_Ks, star_pblum_H)
        stellar_params_lcfit = (star_mass, star_rad, star_teff,
                                star_mag_Ks, star_mag_H,
                                star_pblum_Ks, star_pblum_H)
        
        return stellar_params_all, stellar_params_lcfit
    
    
    def calc_pb_lums(self, mag_Ks, mag_H):
        # Calculate absolute magnitude (subtract distance modulus)
        mag_Ks = mag_Ks - 5. * np.log10(self.dist / 10.)
        mag_H = mag_H - 5. * np.log10(self.dist / 10.)
    
        # Subtract extinction
        mag_Ks = mag_Ks - self.A_Ks
        mag_H = mag_H - self.A_H
    
        # Calculate luminosity
        ## Convert magnitudes into fluxes
        flux_Ks = flux_ref_Ks * (10.**((mag_Ks - 0.03) / -2.5))
        flux_H = flux_ref_H * (10.**((mag_H - 0.03) / -2.5))
    
        ## Calculate passband luminosities
        lum_Ks = flux_Ks * (4. * np.pi * (10. * u.pc)**2.)
        lum_H = flux_H * (4. * np.pi * (10. * u.pc)**2.)
    
        # Return passband luminosity
        return lum_Ks.to(u.solLum), lum_H.to(u.solLum)

