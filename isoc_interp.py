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


class isochrone_mist:
    # Evolution/Atmosphere Models and Extinction Law
    evo_model = evolution.MISTv1()
    atm_func = atmospheres.get_merged_atmosphere
    red_law = reddening.RedLawNoguerasLara18()
    
    filt_list = ['nirc2,Kp', 'nirc2,H']
    
    def __init__(self, age=3.9*(10.**6.), ext=2.63, dist=7.971e3):
        logAge = np.log10(3.9 * (10.**6.), phase=None)
        
        # Create an isochhrone with the given parameters
        self.iso_curAge = synthetic.IsochronePhot(logAge, ext, dist,
                                                  evo_model=evo_model, atm_func=atm_func,
                                                  red_law=red_law, filters=filt_list)
        
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
        
        self.iso_mag_Kp = self.iso_curAge.points['m_nirc2_Kp'][phase_check]
        self.iso_mag_H = self.iso_curAge.points['m_nirc2_H'][phase_check]
        
        self.iso_rad_min = np.min(self.iso_rad).value
        self.iso_rad_max = np.max(self.iso_rad).value
        
        
    def rad_interp(star_rad_interp):
        # Reverse isochrones, if radius not increasing, for numpy interpolation to work
        if self.iso_rad[-1] < self.iso_rad[0]:
            self.iso_mass_init = self.iso_mass_init[::-1]
            self.iso_mass = self.iso_mass[::-1]
            self.iso_rad = self.iso_rad[::-1]
            self.iso_lum = self.iso_lum[::-1]
            self.iso_teff = self.iso_teff[::-1]

            self.iso_mag_Kp = self.iso_mag_Kp[::-1]
            self.iso_mag_H = self.iso_mag_H[::-1]
        
        star_rad = star_rad_interp * u.solRad
        
        star_mass_init = np.interp(star_rad_interp, self.iso_rad, self.iso_mass_init) * u.solMass
        star_mass = np.interp(star_rad_interp, self.iso_rad, self.iso_mass) * u.solMass
        star_lum = np.interp(star_rad_interp, self.iso_rad, self.iso_lum) * u.W
        star_teff = np.interp(star_rad_interp, self.iso_rad, self.iso_teff) * u.K
        star_mag_Kp = np.interp(star_rad_interp, self.iso_rad, self.iso_mag_Kp)
        star_mag_H = np.interp(star_rad_interp, self.iso_rad, self.iso_mag_H)
        
        # Export tuple with all parameters and tuple with only parameters needed for lc fit
        stellar_params_all = (star_mass_init, star_mass, star_rad, star_lum, star_teff, star_mag_Kp, star_mag_H)
        stellar_params_lcfit = (star_mass, star_rad, star_teff, star_mag_Kp, star_mag_H)
        
        return stellar_params_all, stellar_params_lcfit
    
    def mass_init_interp(star_mass_init_interp):
        star_mass_init = star_mass_init_interp * u.solMass
        
        star_mass = np.interp(star_mass_init_interp, self.iso_mass_init, self.iso_mass) * u.solMass
        star_rad = np.interp(star_mass_init_interp, self.iso_mass_init, self.iso_rad) * u.solRad
        star_lum = np.interp(star_mass_init_interp, self.iso_mass_init, self.iso_lum) * u.W
        star_teff = np.interp(star_mass_init_interp, self.iso_mass_init, self.iso_teff) * u.K
        star_mag_Kp = np.interp(star_mass_init_interp, self.iso_mass_init, self.iso_mag_Kp)
        star_mag_H = np.interp(star_mass_init_interp, self.iso_mass_init, self.iso_mag_H)
        
        # Export tuple with all parameters and tuple with only parameters needed for lc fit
        stellar_params_all = (star_mass_init, star_mass, star_rad, star_lum, star_teff, star_mag_Kp, star_mag_H)
        stellar_params_lcfit = (star_mass, star_rad, star_teff, star_mag_Kp, star_mag_H)
        
        return stellar_params_all, stellar_params_lcfit
    

