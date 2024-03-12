# Observations object for phitter
# ---
# Abhimat Gautam

from phoebe import u
from phoebe import c as const
import numpy as np

class observations(object):
    """
    observations is an object to store observables, either observed or modeled.
    Observed observables will typically have uncertainties associated, but
    modeled observables will typically not.
    
    Parameters
    ----------
    obs_times : np.array(dtype=float), default=None
        Observation times. Phitter assumes times are given in MJD.
    obs : np.array(dtype=float), default=None
        Observables. For photometry, phitter assumes values provided in mags.
        For RV, phitter assumes values provided in km/s.
    obs_uncs : np.array(dtype=float), default=None
        Uncertainties on observables, with same units as observables. For
        photometry, phitter assumes values provided in mags. For RV, phitter
        assumes values provided in km/s.
    obs_filts : np.array(dtype=str), default=None
        Filters of each observable, provided as an array of same length as obs.
    obs_types : np.array(dtype=str), default=None
        Observation type of each observable, provided as an array of same length as obs. Possible types are 'phot' or 'rv'.
    
    Attributes
    ----------
    num_obs : int
        Number of total observables in observations object.
    unique_filts : np.array(dtype=str)
        An array of all the unique filters of observables.
    num_filts : int
        Number of unique filters of observables.
    obs_times_phot : np.array(dtype=float)
        obs_times, but only for photometric observations.
    obs_times_rv : np.array(dtype=float)
        obs_times, but only for RV observations.
    obs_phot : np.array(dtype=float)
        obs, but only for photometric observations.
    obs_rv : np.array(dtype=float)
        obs, but only for RV observations.
    obs_uncs_phot : np.array(dtype=float)
        obs_uncs, but only for photometric observations.
    obs_uncs_rv : np.array(dtype=float)
        obs_uncs, but only for RV observations.
    """
    
    obs_times = np.array([])
    obs = np.array([])
    obs_uncs = None
    obs_filts = np.array([])
    obs_types = np.array([])
    
    def __init__(
        self,
        obs_times=None, obs=None, obs_uncs=None,
        obs_filts=None, obs_types=None,
    ):
        if obs_times != None:
            self.set_obs_times(obs_times)
        
        if obs != None:
            self.set_obs(obs, obs_uncs=obs_uncs)
        
        if obs_filts != None:
            self.set_obs_filts(obs_filts)
        
        if obs_types != None:
            self.set_obs_types(obs_types)
        
    def set_obs_times(self, obs_times):
        self.obs_times = obs_times
        
        self.num_obs = len(self.obs_times)
    
    def set_obs(self, obs, obs_uncs=None):
        self.obs = obs
        
        self.obs_uncs = obs_uncs
        
    def set_obs_filts(self, obs_filts):
        self.obs_filts = obs_filts
        
        self.unique_filts = np.unique(self.obs_filts)
        self.num_filts = len(self.unique_filts)
    
    def set_obs_types(self, obs_types):
        self.obs_types = np.char.lower(obs_types)
        
        self.obs_phot_filter = np.where(
            self.obs_types == 'phot'
        )
        
        self.obs_rv_filter = np.where(
            self.obs_types == 'rv'
        )
        
        self.obs_times_phot = self.obs_times[self.obs_phot_filter]
        self.obs_times_rv = self.obs_times[self.obs_rv_filter]
        
        self.num_obs_phot = len(self.obs_times_phot)
        self.num_obs_rv = len(self.obs_times_rv)
        
        self.obs_phot = self.obs[self.obs_phot_filter]
        self.obs_rv = self.obs[self.obs_rv_filter]
        
        if self.obs_uncs != None:
            self.obs_uncs_phot = self.obs_uncs[self.obs_phot_filter]
            self.obs_uncs_rv = self.obs_uncs[self.obs_rv_filter]
        
        self.obs_filts_phot = self.obs_times[self.obs_phot_filter]
        self.obs_filts_rv = self.obs_times[self.obs_rv_filter]
    
    