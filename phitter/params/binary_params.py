# Objects for star parameters

from phoebe import u
import numpy as np
from astropy.modeling import Model
from astropy import modeling

class binary_params(object):
    """
    binary_params is an object to store parameters describing a binary sytem.
    
    Attributes
    ----------
    period : Astropy Quantity, unit:d
        Orbital period of binary in days.
    ecc : float
        Eccentricity of binary orbit.
    inc : Astropy Quantity, unit:deg
        Binary orbit inclination, relative to line of sight, in degrees.
    t0 : float
        Binary system's reference time, t0. Specified in MJDs. For eclipsing
        systems, typically minimum of deepest eclipse.
    """
    
    period = 0. * u.d
    ecc = 0.
    inc = 90. * u.deg
    t0 = 48546.0
    rv_com = 0.0 * u.km / u.s
    
    def __init__(self):
        return

class binary_params_model(Model):
    """
    Model object that returns a binary_params object, providing Model Parameters
    that can be returned as part of binary_params
    """
    
    period = modeling.Parameter(unit=u.d, min=0.0)
    ecc = modeling.Parameter(min=0.0, max=1.0)
    inc = modeling.Parameter(unit=u.deg, min=0.0, max=180.0)
    t0 = modeling.Parameter()
    rv_com = modeling.Parameter(unit=u.km / u.s)
    
    outputs = ('binary_params',)
    
    @staticmethod
    def evaluate(period, ecc, inc, t0, rv_com):
        # Check if Parameters in keywords are numpy arrays,
        # pull out first value if so
        if type(period.value) == np.ndarray:
            period = period[0]
        
        if type(ecc) == np.ndarray:
            ecc = ecc[0]
        
        if type(inc.value) == np.ndarray:
            inc = inc[0]
        
        if type(t0) == np.ndarray:
            t0 = t0[0]
        
        if type(rv_com.value) == np.ndarray:
            rv_com = rv_com[0]
        
        # Create binary params object
        bin_params_obj = binary_params()
        
        bin_params_obj.period = period
        bin_params_obj.ecc = ecc
        bin_params_obj.inc = inc
        bin_params_obj.t0 = t0
        bin_params_obj.rv_com = rv_com
        
        return bin_params_obj,