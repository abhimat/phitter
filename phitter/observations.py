# Observations object for phitter
# ---
# Abhimat Gautam

from phoebe import u
from phoebe import c as const
import numpy as np

class observations(object):
    """
    observations is an object to store observables, either observed or modeled
    """
    
    obs_times = np.array([])
    obs = np.array([])
    obs_filts = np.array([])
    obs_types = np.array([])
    
    def __init__(
        self,
        obs_times=None, obs=None,
        obs_filts=None, obs_types=None,
    ):
        if obs_times != None:
            self.set_obs_times(obs_times)
        
        if obs != None:
            self.set_obs(obs)
        
        if obs_filts != None:
            self.set_obs_filts(obs_filts)
        
        if obs_types != None:
            self.set_obs_types(obs_types)
        
    def set_obs_times(self, obs_times):
        self.obs_times = obs_times
        
        self.num_obs = len(self.obs_times)
    
    def set_obs(self, obs):
        self.obs = obs
    
    def set_obs_filts(self, obs_filts):
        self.obs_filts = obs_filts
        
        self.unique_filts = np.unique(self.obs_filts)
        self.num_filts = len(self.unique_filts)
    
    def set_obs_types(self, obs_types):
        self.obs_types = obs_types
        
        self.obs_phot_filter = np.where(
            self.obs_types == 'phot'
        )
        
        self.obs_rv_filter = np.where(
            self.obs_types == 'rv'
        )
    
    