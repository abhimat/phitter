#!/usr/bin/env python

# Isochrone interpolation functions,
# using isochrones generated with PopStar
# ---
# Abhimat Gautam

from spisea import synthetic, evolution, atmospheres, reddening
from pysynphot import spectrum
from phitter import filters
from phitter.params.star_params import star_params, stellar_params_obj
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
class bb_stellar_params(stellar_params_obj):
    def __init__(
        self, ext_Ks=2.63, dist=7.971e3,
        filts_list=[kp_filt, h_filt],
        ext_law='NL18',
    ):
        # Call parent stellar_params_obj to initialize
        super().__init__(
            ext_Ks=ext_Ks, dist=dist,
            filts_list=filts_list,
            ext_law=ext_law,
        )
        
        # Define atmosphere
        self.bb_atm_func = atmospheres.get_bb_atmosphere
        
        return
    
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
        
        # Create star params object for output
        star_params_obj = star_params()
        star_params_obj.mass_init = mass.to(u.solMass)
        star_params_obj.mass = mass.to(u.solMass)
        star_params_obj.rad = rad.to(u.solRad)
        star_params_obj.lum = bb_lum.to(u.solLum)
        star_params_obj.teff = teff.to(u.K)
        star_params_obj.logg = logg
        
        star_params_obj.filts = self.filts_list
        star_params_obj.mags = filt_mags
        star_params_obj.mags_abs = filt_absMags
        star_params_obj.pblums = filt_pblums
        
        return star_params_obj
    
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
    
