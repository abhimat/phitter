# Observables object for phitter
# ---
# Abhimat Gautam

from phoebe import u
from phoebe import c as const
import numpy as np

class observables(object):
    """
    observables is an object to store observables, either observed or modeled.
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
    obs_filts : list of filters, default=None
        Filters of each observable, provided as a list of filter objects of same
        length as obs.
    obs_types : np.array(dtype=str), default=None
        Observation type of each observable, provided as an array of same length
        as obs. Possible types are 'phot' or 'rv'.
    
    Attributes
    ----------
    num_obs : int
        Number of total observables in observations object.
    unique_filts : list of filters
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
    obs_uncs = np.array([])
    obs_filts = np.array([])
    obs_types = np.array([])
    
    num_obs = 0
    
    def __init__(
        self,
        obs_times=None, obs=None, obs_uncs=None,
        obs_filts=None, obs_types=None,
    ):
        if obs_times is not None:
            self.set_obs_times(obs_times)
        
        if obs is not None:
            self.set_obs(obs, obs_uncs=obs_uncs)
        
        if obs_filts is not None:
            self.set_obs_filts(obs_filts)
        
        if obs_types is not None:
            self.set_obs_types(obs_types)
        
        if obs_filts is not None and obs_types is not None:
            self._make_filt_search_filters()
        
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
        
        self.obs_rv_filter = np.where(np.logical_or(
            self.obs_types == 'rv_pri',
            self.obs_types == 'rv_sec',
        ))
        
        self.obs_rv_pri_filter = np.where(
            self.obs_types == 'rv_pri',
        )
        self.obs_rv_sec_filter = np.where(
            self.obs_types == 'rv_sec',
        )
        
        if len(self.obs_times) > 0:
            self.obs_times_phot = self.obs_times[self.obs_phot_filter]
            self.obs_times_rv = self.obs_times[self.obs_rv_filter]
            self.obs_times_rv_pri = self.obs_times[self.obs_rv_pri_filter]
            self.obs_times_rv_sec = self.obs_times[self.obs_rv_sec_filter]
            self.obs_times_rv_unique = np.unique(self.obs_times_rv)
                    
            self.num_obs_phot = len(self.obs_times_phot)
            self.num_obs_rv = len(self.obs_times_rv)
            self.num_obs_rv_pri = len(self.obs_times_rv_pri)
            self.num_obs_rv_sec = len(self.obs_times_rv_sec)
        
        if len(self.obs) > 0:
            self.obs_phot = self.obs[self.obs_phot_filter]
            self.obs_rv = self.obs[self.obs_rv_filter]
            self.obs_rv_pri = self.obs[self.obs_rv_pri_filter]
            self.obs_rv_sec = self.obs[self.obs_rv_sec_filter]
        
        if self.obs_uncs is not None and len(self.obs_uncs) > 0:
            self.obs_uncs_phot = self.obs_uncs[self.obs_phot_filter]
            self.obs_uncs_rv = self.obs_uncs[self.obs_rv_filter]
            self.obs_uncs_rv_pri = self.obs_uncs[self.obs_rv_pri_filter]
            self.obs_uncs_rv_sec = self.obs_uncs[self.obs_rv_sec_filter]
        
        if len(self.obs_filts) > 0:
            self.obs_filts_phot = self.obs_filts[self.obs_phot_filter]
            self.obs_filts_rv = self.obs_filts[self.obs_rv_filter]
            
            self.unique_filts_phot = np.unique(self.obs_filts_phot)
            self.unique_filts_rv = np.unique(self.obs_filts_rv)
            
            self.num_filts_phot = len(self.unique_filts_phot)
            self.num_filts_rv = len(self.unique_filts_rv)
    
    def _make_filt_search_filters(self):
        """Private function to make search filters for every filter
        """
        
        self.phot_filt_filters = {}
        self.rv_filt_filters = {}
        
        if self.num_filts_phot > 0:
            for filt in self.unique_filts_phot:
                search_filter = np.where(np.logical_and(
                    self.obs_types == 'phot',
                    self.obs_filts == filt,
                ))
                
                self.phot_filt_filters[filt] = search_filter
        
        if self.num_filts_rv > 0:
            for filt in self.unique_filts_rv:
                search_filter = np.where(np.logical_and(
                    np.logical_or(
                        self.obs_types == 'rv_pri',
                        self.obs_types == 'rv_sec',
                    ),
                    self.obs_filts == filt,
                ))
                
                self.rv_filt_filters[filt] = search_filter
        
        return
