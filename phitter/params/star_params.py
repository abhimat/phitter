# Objects for star parameters

from phoebe import u
from phoebe import c as const
import numpy as np
from spisea import synthetic, reddening
from phitter import filters

class star_params(object):
    """
    star_params is an object to store stellar parameters. These parameters can
    be interpolated from an isochrone, from a black body function, or
    constructed from scratch.
    
    Attributes
    ----------
    mass_init : Astropy Quantity, unit:solMass
        Initial stellar mass in solar masses
    mass : Astropy Quantity, unit:solMass
        Current stellar mass in solar masses
    rad : Astropy Quantity, unit:solRad
        Current stellar radius in solar radii
    lum : Astropy Quantity, unit:solLum
        Current stellar luminosity in solar luminosities
    teff : Astropy Quantity, unit:K
        Current effective temperature of star in Kelvin
    logg : float
        Current surface gravity of star, specified as a unitless quantity as
        log_10 (g / [1 * cm s^-2])
    filts : list_like
        List of phitter filters / passbands that mags and pblums are generated for.
    mags : array_like(dtype=float)
        Array of apparent magnitude in filters / passbands being used.
    mags_abs : array_like(dtype=float)
        Array of absolute magnitude in filters / passbands being used.
    pblums : array_like(dtype=Astropy Quantity)
        Array of passband luminosities in filters / passbands being used, each
        in units of solar luminosities. Passband luminosity in a filter /
        passband is defined as the luminosity of the star only in that passband.
    """
    
    mass_init = 0. * u.solMass
    mass = 0. * u.solMass
    rad = 0. * u.solRad
    lum = 0. * u.solLum
    teff = 0. * u.K
    logg = 0.
    
    filts = []
    mags = np.empty()
    mags_abs = np.empty()
    pblums = np.empty() * u.solLum
    
    def __init__(self):
        return

# Filters for default filter list
ks_filt_info = synthetic.get_filter_info('naco,Ks')
v_filt_info = synthetic.get_filter_info('ubv,V')

flux_ref_Ks = ks_filt_info.flux0 * (u.erg / u.s) / (u.cm**2.)
flux_ref_V = v_filt_info.flux0 * (u.erg / u.s) / (u.cm**2.)

kp_filt = filters.nirc2_kp_filt()
h_filt = filters.nirc2_h_filt()

red_law_options = {
    'NL18': reddening.RedLawNoguerasLara18(),
    'F11': reddening.RedLawFritz11(),
}

class stellar_params_obj(object):
    """
    Base stellar parameters object. Provides common functionality across objects
    used for obtaining star parameters.
    """
    
    def __init__(
        self, ext_Ks=2.63, dist=7.971e3,
        filts_list=[kp_filt, h_filt],
        ext_law='NL18',
    ):
        # Define extinction and distance
        self.A_Ks = ext_Ks
        self.dist = dist * u.pc
        
        # Specify filters and get filter information
        self.filts_list = filts_list
        self.num_filts = len(self.filts_list)
        
        self._calc_filts_info()
        
        self.red_law = red_law_options[ext_law]
        
        return
        
    def _calc_filts_info(self):
        self.filts_info = []
        self.filts_flux_ref = np.empty(self.num_filts) *\
            (u.erg / u.s) / (u.cm**2.)
        
        for cur_filt_index in range(self.num_filts):
            cur_filt = self.filts_list[cur_filt_index]
            
            cur_filt_info = cur_filt.filt_info
            self.filts_info.append(cur_filt_info)
            
            cur_filt_flux_ref = cur_filt.flux_ref_filt
            self.filts_flux_ref[cur_filt_index] = cur_filt_flux_ref
    
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
        