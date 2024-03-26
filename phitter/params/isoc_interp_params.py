#!/usr/bin/env python

# Isochrone interpolation functions,
# using isochrones generated with PopStar
# ---
# Abhimat Gautam

from spisea import synthetic, evolution, atmospheres, reddening
from spisea.imf import imf, multiplicity
from pysynphot import spectrum
from phitter import filters
from phitter.params.star_params import star_params, stellar_params_obj
from phoebe import u
from phoebe import c as const
import numpy as np
import matplotlib.pyplot as plt

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

# Dictionaries to map SPISEA specifications
atm_funcs = {
    'merged': atmospheres.get_merged_atmosphere,
    'castelli': atmospheres.get_castelli_atmosphere,
    'phoenix': atmospheres.get_phoenixv16_atmosphere,
}

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

# Object to get synthetic magnitudes for blackbody objects
class isoc_mist_stellar_params(stellar_params_obj):    
    """
    stellar_params class, to derive stellar parameters from a MIST isochrone.
    
    Parameters
    ----------
    age : float, default=4e6
        Age of the isochrone, specified in years. Default: 4e6 (4 Myr old).
    met : float, default=0.0
    ext_Ks : float, default=2.63
        Extinction of stellar parameters object in Ks band. Used for calculating
        synthetic photometry.
    dist : Astropy Quantity, length units, default=7.971e3*u.pc
        Distance to the isochrone, used for calculating synthetic photometry.
    filts_list : [filter], default=[filters.nirc2_kp_filt(), filters.nirc2_h_filt()]
        List of filters to compute synthetic photometry.
    ext_law : str, default='NL18'
        Choice of extinction law to use when computing synthetic photometry.
        Default is 'NL18', corresponding to Nogueras-Lara+ 2018.
    use_atm_func : str, default='merged'
        Atmospheric function to use for calculating synthetic photometry.
        Options are 'merged' for merged atmospheres from SPISEA, 'castelli' for
        Castelli & Kurucz (2004) atmospheres, or 'phoenix' for Phoenix v16
        (Husser et al., 2013) atmospheres.
    phase : str, default=None
        If specified, only select stars from a specific stellar evolution phase.
        Options are 'PMS', 'MS', 'RGB', 'CHeB', 'EAGB', 'TPAGB', 'postAGB', or
        'WR'. If phase is None, then all stellar evolution phases are included.
        Default: None (all stellar evolution phases).
    """
    
    def __init__(
        self, age=4e6, met=0.0,
        use_atm_func='merged', phase=None,
        *args, **kwargs,
    ):
        # Call parent stellar_params_obj to initialize
        super(isoc_mist_stellar_params, self).__init__(
            *args, **kwargs,
        )
        
        log_age = np.log10(age)
        self.log_age = log_age
        self.met = met
        
        # Evolution and Atmosphere Models
        evo_model = evolution.MISTv1()
        atm_func = atm_funcs[use_atm_func]
        
        # Create an isochrone with the given parameters        
        self.iso_curAge = synthetic.IsochronePhot(
            self.log_age, self.A_Ks, self.dist.to(u.pc).value,
            evo_model=evo_model,
            atm_func=atm_func,
            red_law=self.red_law,
            metallicity=self.met,
            filters=self.spisea_filts_list,
        )
        
        # Create another isochrone for calculating absolute mags and
        # passband luminosities, needed for PHOEBE
        self.iso_absMag = synthetic.IsochronePhot(
            self.log_age, 0.0, 10.0,
            evo_model=evo_model,
            atm_func=atm_func,
            red_law=self.red_law,
            metallicity=self.met,
            filters=self.spisea_filts_list,
        )
        
        # Save out the specific stellar parameter columns needed
        ## If needing specific phase, draw it out before saving
        if phase is not None:
            phase_check = np.where(
                self.iso_curAge.points['phase'] == mist_phase_dict[phase]
            )
        else:
            phase_check = np.where(self.iso_curAge.points['phase'] >= -1)
        
        self.iso_mass_init = (self.iso_curAge.points['mass'][phase_check]).to(u.solMass).value
        self.iso_mass = (self.iso_curAge.points['mass_current'][phase_check]).to(u.solMass).value
        self.iso_rad = (self.iso_curAge.points['R'][phase_check]).to(u.solRad).value
        self.iso_lum = (self.iso_curAge.points['L'][phase_check]).to(u.W).value
        self.iso_teff = (self.iso_curAge.points['Teff'][phase_check]).to(u.K).value
        self.iso_logg = self.iso_curAge.points['logg'][phase_check]
        
        self.iso_mag_filts = {}
        
        for filt in self.filts_list:
            self.iso_mag_filts[filt] = self.iso_curAge.points['m_' + filt.spisea_name][phase_check]
        
        ## Stellar parameters from the absolute magnitude isochrones
        self.iso_absMag_mass_init = (self.iso_absMag.points['mass'][phase_check]).to(u.solMass).value
        self.iso_absMag_mass = (self.iso_absMag.points['mass_current'][phase_check]).to(u.solMass).value
        self.iso_absMag_rad = (self.iso_absMag.points['R'][phase_check]).to(u.solRad).value
        
        self.iso_absMag_filts = {}
        
        for filt in self.filts_list:
            self.iso_absMag_filts[filt] = self.iso_absMag.points['m_' + filt.spisea_name][phase_check]
        
        ## Maximum bounds on the radius in isochrone
        self.iso_rad_min = np.min(self.iso_rad)
        self.iso_rad_max = np.max(self.iso_rad)
        
        ## Maximum bounds on the initial mass in isochrone
        self.iso_mass_init_min = np.min(self.iso_mass_init)
        self.iso_mass_init_max = np.max(self.iso_mass_init)
        
        return
    
    def interp_star_params_mass_init(self, mass_init):
        """
        Interpolate stellar parameters from the isochrone, given an initial mass
        for the star.
        
        Parameters
        ----------
        mass_init : float
            Initial stellar mass as float, in units of solar masses.
        
        Returns
        -------
        star_params
            star_params object returned, with stellar parameters interpolated
            from the MIST isochrone.
        """
        
        # Create star params object for output
        star_params_obj = star_params()
        
        # Interpolate stellar parameters and set
        star_params_obj.mass_init = mass_init * u.solMass
        star_params_obj.mass = np.interp(
            mass_init, self.iso_mass_init, self.iso_mass,
        ) * u.solMass
        star_params_obj.rad = np.interp(
            mass_init, self.iso_mass_init, self.iso_rad,
        ) * u.solRad
        star_params_obj.lum = np.interp(
            mass_init, self.iso_mass_init, self.iso_lum,
        ) * u.W
        star_params_obj.teff = np.interp(
            mass_init, self.iso_mass_init, self.iso_teff,
        ) * u.K
        star_params_obj.logg = np.interp(
            mass_init, self.iso_mass_init, self.iso_logg,
        )
        
        # Interpolate mags for every filter
        filt_mags = np.empty(self.num_filts)
        filt_absMags = np.empty(self.num_filts)
        
        for filt_index, filt in enumerate(self.filts_list):
            filt_mags[filt_index] = np.interp(
                mass_init, self.iso_mass_init,
                self.iso_mag_filts[filt],
            )
            
            filt_absMags[filt_index] = np.interp(
                mass_init, self.iso_mass_init,
                self.iso_absMag_filts[filt],
            )
        
        # Calculate passband luminosities
        filt_pblums = self.calc_pblums(filt_absMags)
        
        # Set photometric info
        star_params_obj.filts = self.filts_list
        star_params_obj.mags = filt_mags
        star_params_obj.mags_abs = filt_absMags
        star_params_obj.pblums = filt_pblums
        
        return star_params_obj
    
    def interp_star_params_rad(self, rad):
        """
        Interpolate stellar parameters from the isochrone, given a radius for
        the star.
        
        Parameters
        ----------
        rad : float
            Stellar radius as float, in units of solar radii.
        
        Returns
        -------
        star_params
            star_params object returned, with stellar parameters interpolated
            from the MIST isochrone.
        """
        
        # In order for isochrone interpolation to work with numpy,
        # radius has to be increasing. Flip isochrone if not increasing.
        if self.iso_rad[-1] < self.iso_rad[0]:
            self._flip_isochrone()
            
        # Create star params object for output
        star_params_obj = star_params()
        
        # Interpolate stellar parameters and set
        star_params_obj.mass_init = np.interp(
            rad, self.iso_rad, self.iso_mass_init,
        ) * u.solMass
        star_params_obj.mass = np.interp(
            rad, self.iso_rad, self.iso_mass,
        ) * u.solMass
        star_params_obj.rad = rad * u.solRad
        star_params_obj.lum = np.interp(
            rad, self.iso_rad, self.iso_lum,
        ) * u.W
        star_params_obj.teff = np.interp(
            rad, self.iso_rad, self.iso_teff,
        ) * u.K
        star_params_obj.logg = np.interp(
            rad, self.iso_rad, self.iso_logg,
        )
        
        # Interpolate mags for every filter
        filt_mags = np.empty(self.num_filts)
        filt_absMags = np.empty(self.num_filts)
        
        for filt_index, filt in enumerate(self.filts_list):
            filt_mags[filt_index] = np.interp(
                rad, self.iso_rad,
                self.iso_mag_filts[filt],
            )
            
            filt_absMags[filt_index] = np.interp(
                rad, self.iso_rad,
                self.iso_absMag_filts[filt],
            )
        
        # Calculate passband luminosities
        filt_pblums = self.calc_pblums(filt_absMags)
        
        # Set photometric info
        star_params_obj.filts = self.filts_list
        star_params_obj.mags = filt_mags
        star_params_obj.mags_abs = filt_absMags
        star_params_obj.pblums = filt_pblums
        
        return star_params_obj
    
    def interp_star_params_mass(self, mass):
        """
        Interpolate stellar parameters from the isochrone, given a mass for
        the star.
        
        Parameters
        ----------
        mass : float
            Stellar mass as float, in units of solar masses.
        
        Returns
        -------
        star_params
            star_params object returned, with stellar parameters interpolated
            from the MIST isochrone.
        """
        
        # In order for isochrone interpolation to work with numpy,
        # mass has to be increasing. Flip isochrone if not increasing.
        if self.iso_mass[-1] < self.iso_mass[0]:
            self._flip_isochrone()
        
        # Create star params object for output
        star_params_obj = star_params()
        
        # Interpolate stellar parameters and set
        star_params_obj.mass_init = np.interp(
            mass, self.iso_mass, self.iso_mass_init,
        ) * u.solMass
        star_params_obj.mass = mass * u.solMass
        star_params_obj.rad = np.interp(
            mass, self.iso_mass, self.iso_rad,
        )
        star_params_obj.lum = np.interp(
            mass, self.iso_mass, self.iso_lum,
        ) * u.W
        star_params_obj.teff = np.interp(
            mass, self.iso_mass, self.iso_teff,
        ) * u.K
        star_params_obj.logg = np.interp(
            mass, self.iso_mass, self.iso_logg,
        )
        
        # Interpolate mags for every filter
        filt_mags = np.empty(self.num_filts)
        filt_absMags = np.empty(self.num_filts)
        
        for filt_index, filt in enumerate(self.filts_list):
            filt_mags[filt_index] = np.interp(
                mass, self.iso_mass,
                self.iso_mag_filts[filt],
            )
            
            filt_absMags[filt_index] = np.interp(
                mass, self.iso_mass,
                self.iso_absMag_filts[filt],
            )
        
        # Calculate passband luminosities
        filt_pblums = self.calc_pblums(filt_absMags)
        
        # Set photometric info
        star_params_obj.filts = self.filts_list
        star_params_obj.mags = filt_mags
        star_params_obj.mags_abs = filt_absMags
        star_params_obj.pblums = filt_pblums
        
        return star_params_obj
    
    def _flip_isochrone(self):
        """Flip isochrone parameter lists, if needed for interpolation"""
        self.iso_mass_init = self.iso_mass_init[::-1]
        self.iso_mass = self.iso_mass[::-1]
        self.iso_rad = self.iso_rad[::-1]
        self.iso_lum = self.iso_lum[::-1]
        self.iso_teff = self.iso_teff[::-1]
        self.iso_logg = self.iso_logg[::-1]
        
        self.iso_absMag_mass_init = self.iso_absMag_mass_init[::-1]
        self.iso_absMag_mass = self.iso_absMag_mass[::-1]
        self.iso_absMag_rad = self.iso_absMag_rad[::-1]
        
        for filt in self.filts_list:
            self.iso_mag_filts[filt] = self.iso_mag_filts[filt][::-1]
            self.iso_absMag_filts[filt] = self.iso_absMag_filts[filt][::-1]