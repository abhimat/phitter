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
