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
    rv_com : Astropy Quantity, unit:km/s
        Binary system's center of mass velocity.
    arg_per0 : Astropy Quantity, unit:deg
        Binary's argument of periastron, in degrees, at time t0.
    long_an : Astropy Quantity, unit:deg
        Binary orbit's longitude of ascending node.
    """
    
    period = 0. * u.d
    ecc = 0.
    inc = 90. * u.deg
    t0 = 48546.0
    rv_com = 0.0 * u.km / u.s
    arg_per0 = 0. * u.deg
    long_an = 0. * u.deg
    
    def __init__(self):
        return
