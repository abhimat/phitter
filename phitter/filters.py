# Objects to make working with filters easier

from phoebe import u
from phoebe import c as const

import numpy as np
from spisea import synthetic

lambda_Ks = 2.18e-6 * u.m

class filter(object):
    def __init__(self):
        
        # Filter properties
        self.lambda_filt = 0.0 * u.m
        self.dlambda_filt = 0.0 * u.m
        
        self.filt_info = None
        
        self.flux_ref_filt = 0.0 * (u.erg / u.s) / (u.cm**2.)
        
        return
    
    def calc_isoc_filt_ext(isoc_Ks_ext, ext_alpha):
        isoc_filt_ext = isoc_Ks_ext *\
                        (lambda_Ks / self.lambda_filt)**ext_alpha
    
class naco_ks_filt(filter):
    def __init__(self):
        
        self.filter_name = 'naco,Ks'
        
        # Filter properties
        self.lambda_filt = 2.18e-6 * u.m
        self.dlambda_filt = 0.35e-6 * u.m
        
        self.filt_info = synthetic.get_filter_info(self.filter_name)
        
        self.flux_ref_filt = self.filt_info.flux0 * (u.erg / u.s) / (u.cm**2.)
        
        return

class nirc2_lp_filt(filter):
    def __init__(self):
        
        self.filter_name = 'nirc2,Lp'
        
        # Filter properties
        self.lambda_filt = 3.776e-6 * u.m
        self.dlambda_filt = 0.700e-6 * u.m
        
        self.filt_info = synthetic.get_filter_info(self.filter_name)
        
        self.flux_ref_filt = self.filt_info.flux0 * (u.erg / u.s) / (u.cm**2.)
        
        return

class nirc2_kp_filt(filter):
    def __init__(self):
        
        self.filter_name = 'nirc2,Kp'
        
        # Filter properties
        self.lambda_filt = 2.124e-6 * u.m
        self.dlambda_filt = 0.351e-6 * u.m
        
        self.filt_info = synthetic.get_filter_info(self.filter_name)
        
        self.flux_ref_filt = self.filt_info.flux0 * (u.erg / u.s) / (u.cm**2.)
        
        return

class nirc2_h_filt(filter):
    def __init__(self):
        
        self.filter_name = 'nirc2,H'
        
        # Filter properties
        self.lambda_filt = 1.633e-6 * u.m
        self.dlambda_filt = 0.296e-6 * u.m
        
        self.filt_info = synthetic.get_filter_info(self.filter_name)
        
        self.flux_ref_filt = self.filt_info.flux0 * (u.erg / u.s) / (u.cm**2.)
        
        return