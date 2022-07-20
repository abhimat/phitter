#!/usr/bin/env python

# Isochrone interpolation functions,
# using isochrones generated with PopStar
# ---
# Abhimat Gautam

from spisea import synthetic, evolution, atmospheres, reddening
from pysynphot import spectrum
from . import filters
from phoebe import u
from phoebe import c as const
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

# Filter properties
lambda_Ks = 2.18e-6 * u.m
dlambda_Ks = 0.35e-6 * u.m

# Reference fluxes, calculated with PopStar
## Vega magnitudes (m_Vega = 0.03)
ks_filt_info = synthetic.get_filter_info('naco,Ks')
v_filt_info = synthetic.get_filter_info('ubv,V')

flux_ref_Ks = ks_filt_info.flux0 * (u.erg / u.s) / (u.cm**2.)
flux_ref_V = v_filt_info.flux0 * (u.erg / u.s) / (u.cm**2.)

# Filters for default filter list
kp_filt = filters.nirc2_kp_filt()
h_filt = filters.nirc2_h_filt()

# Object to get synthetic magnitudes for blackbody objects
class bb_stellar_params(object):
    def __init__(self, ext=2.63, dist=7.971e3,
                 filts_list=[kp_filt, h_filt]):
        # Define extinction and distance
        self.A_Ks = ext
        self.dist = dist * u.pc
        
        # Specify filters and get filter information
        self.filts_list = filts_list
        self.num_filts = len(self.filts_list)
        
        self.filts_info = []
        self.filts_flux_ref = np.empty(self.num_filts) *\
                                  (u.erg / u.s) / (u.cm**2.)
        for cur_filt_index in range(self.num_filts):
            cur_filt = self.filts_list[cur_filt_index]
            
            cur_filt_info = cur_filt.filt_info
            self.filts_info.append(cur_filt_info)
            
            cur_filt_flux_ref = cur_filt.flux_ref_filt
            self.filts_flux_ref[cur_filt_index] = cur_filt_flux_ref
        
        # Define atmosphere and reddening functions
        self.bb_atm_func = atmospheres.get_bb_atmosphere
        self.red_law = reddening.RedLawNoguerasLara18()
    
    def calc_stellar_params(self, mass, rad, teff):
        # Calculate surface gravity
        grav = (const.G * mass) / (rad**2)
        logg = np.log10(grav.cgs.value)
        
        # Calculate total luminosity
        bb_flux = const.sigma_sb * (teff ** 4.)
        bb_surf_area = 4. * np.pi * (rad ** 2.)
        bb_lum = bb_flux * bb_surf_area
        
        # Calculate magnitudes
        filt_mags, filt_absMags = self.get_bb_mags(teff, rad)
        
        # Calculate passband luminosities
        filt_pblums = self.calc_pblums(filt_absMags)
        
        # Export tuple with all parameters and tuple with only parameters needed for lc fit
        stellar_params_all = (mass.to(u.solMass), mass.to(u.solMass),
                              rad.to(u.solRad), bb_lum.to(u.solLum),
                              teff.to(u.K), logg,
                              filt_mags, filt_pblums)
        stellar_params_lcfit = (mass.to(u.solMass), rad.to(u.solRad),
                                teff.to(u.K), logg,
                                filt_mags, filt_pblums)
        
        return stellar_params_all, stellar_params_lcfit
    
    def get_bb_mags(self, bb_temp, bb_rad, diagnostic_plot=False):
        if diagnostic_plot:
            fig = plt.figure(figsize=(8,4))
            ax1 = fig.add_subplot(1, 1, 1)
        
        bb_atm = self.bb_atm_func(temperature=bb_temp.to(u.K).value)
        
        if diagnostic_plot:
            ax1.plot(bb_atm.wave, bb_atm.flux,
                     lw=3, ls='--',
                     color='C0', label='original bbatm')
        
        # Trim wavelength range down to JHKL range (0.5 - 5.2 microns)
        wave_range=[5000, 52000]
        bb_atm = spectrum.trimSpectrum(bb_atm, wave_range[0], wave_range[1])
        
        if diagnostic_plot:
            ax1.plot(bb_atm.wave, bb_atm.flux,
                     color='C0', label='trimmed, unreddened')
        
        # Convert into flux observed at Earth (unreddened)
        # (in erg s^-1 cm^-2 A^-1)
        bb_absMag_atm = bb_atm * ((bb_rad / (10. * u.pc)).to(1).value)**2
        bb_atm = bb_atm * ((bb_rad / self.dist).to(1).value)**2
        
        # Redden the spectrum
        red = self.red_law.reddening(self.A_Ks).resample(bb_atm.wave) 
        bb_atm *= red
        
        if diagnostic_plot:
            ax1.plot(bb_atm.wave, bb_atm.flux,
                     color='C1', label='trimmed, reddened')
        
        if diagnostic_plot:
            ax1.legend()
            ax1.set_xlabel(bb_atm.waveunits)
            ax1.set_ylabel(bb_atm.fluxunits)
            
            ax1.set_yscale('log')
            
            fig.tight_layout()
            fig.savefig('./diagnostic_bb_plot.pdf')
            fig.savefig('./diagnostic_bb_plot.png',
                        dpi=200)
            plt.close(fig)
        
        
        # Calculate mags and absolute Mags for each filter
        filt_bb_mags = np.empty(self.num_filts)
        filt_bb_absMags = np.empty(self.num_filts)
        
        for cur_filt_index in range(self.num_filts):
            cur_filt = self.filts_list[cur_filt_index]
            cur_filt_info = self.filts_info[cur_filt_index]
            
            # Make synthetic photometry
            cur_filt_mag = synthetic.mag_in_filter(bb_atm,
                                                   cur_filt_info)
            filt_bb_mags[cur_filt_index] = cur_filt_mag
            
            cur_filt_absMag = synthetic.mag_in_filter(bb_absMag_atm,
                                                      cur_filt_info)
            filt_bb_absMags[cur_filt_index] = cur_filt_absMag
        
        return filt_bb_mags, filt_bb_absMags
    
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
