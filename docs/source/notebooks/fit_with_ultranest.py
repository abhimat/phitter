#!/usr/bin/env python

# Turn off parallelisation in phoebe
# Needs to be done *before* phoebe is imported
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PHOEBE_ENABLE_MPI"] = "FALSE"

from phitter import observables, filters
from phitter.params import star_params, binary_params, isoc_interp_params
from phitter.calc import model_obs_calc, phot_adj_calc, rv_adj_calc
from phitter.fit import likelihood, prior

import numpy as np

from phoebe import u
from phoebe import c as const
import matplotlib as mpl
import matplotlib.pyplot as plt

import pickle
from astropy.table import Table

import ultranest
import ultranest.stepsampler

from mpi4py import MPI

# Set up filters
filter_153m = filters.hst_f153m_filt()
filter_127m = filters.hst_f127m_filt()

# Read observations data
with open('./mock_obs_table.pkl', 'rb') as in_file:
    obs_table = pickle.load(in_file)

# Set up observables objects
# Model observables object, which only contains times and types of observations
model_observables = observables.observables(
    obs_times=obs_table['obs_times'].data,
    obs_filts=obs_table['obs_filts'].data, obs_types=obs_table['obs_types'].data,
)

# An observables object for the observations, used when computing likelihoods
observations = observables.observables(
    obs_times=obs_table['obs_times'].data, obs=obs_table['obs'].data, obs_uncs=obs_table['obs_uncs'].data,
    obs_filts=obs_table['obs_filts'].data, obs_types=obs_table['obs_types'].data,
)

# Make stellar parameters object
isoc_stellar_params_obj = isoc_interp_params.isoc_mist_stellar_params(
    age=8e9,
    met=0.0,
    use_atm_func='merged',
    phase='RGB',
    ext_Ks=2.2,
    dist=8e3*u.pc,
    filts_list=[filter_153m, filter_127m],
    ext_law='NL18',
)

# Make binary params object
bin_params = binary_params.binary_params()

# Set up a binary model object
binary_model_obj = model_obs_calc.binary_star_model_obs(
    model_observables,
    use_blackbody_atm=False,
    print_diagnostics=False,
)

# Set up likelihood object for fitting parameters
log_like_obj = likelihood.log_likelihood_chisq(
    observations
)

# Ultranest log like function
def un_evaluate(model_params, print_like=False):
    (
        star1_radius,
        star2_radius,
        bin_period,
        bin_inc,
        bin_t0,
        bin_rv_com,
        ext_153m,
        ext_alpha,
    ) = model_params
    
    # Obtain stellar params by interpolating along the isochrone
    star1_params = isoc_stellar_params_obj.interp_star_params_rad(
        star1_radius,
    )
    star2_params = isoc_stellar_params_obj.interp_star_params_rad(
        star2_radius,
    )
    
    # Set binary params
    bin_params.period = bin_period * u.d
    bin_params.inc = bin_inc * u.deg
    bin_params.t0 = bin_t0
    
    # Run binary model
    modeled_observables = binary_model_obj.compute_obs(
        star1_params, star2_params, bin_params,
        num_triangles=300,
    )
    
    # Check for situation where binary model fails
    # (i.e., unphysical conditions not able to be modeled)
    if np.isnan(modeled_observables.obs_times[0]):
        return -1e300
    
    # Apply distance modulus
    # (We're assuming we know the distance, but this can be a fit parameter as well)
    modeled_observables = phot_adj_calc.apply_distance_modulus(
        modeled_observables,
        8e3*u.pc,
    )
    
    # Apply extinction
    modeled_observables = phot_adj_calc.apply_extinction(
        modeled_observables,
        2.2, filter_153m,
        ext_153m,
        isoc_red_law='NL18',
        ext_alpha=ext_alpha,
    )
    
    # Add RV center of mass velocity
    modeled_observables = rv_adj_calc.apply_com_velocity(
        modeled_observables,
        bin_rv_com * u.km / u.s,
    )
    
    # Compute and return log likelihood
    log_like = log_like_obj.evaluate(modeled_observables)
    
    return log_like

# Set up priors
star1_rad_prior = prior.uniform_prior(10.0, 25.0)
star2_rad_prior = prior.uniform_prior(8.0, 15.0)

bin_period_prior = prior.uniform_prior(24.0, 26.0)
bin_inc_prior = prior.uniform_prior(0.0, 180.0)
bin_t0_prior = prior.uniform_prior(53_795.0, 53_805.0)
bin_rv_com_prior = prior.uniform_prior(100., 200.)

ext_f153m_prior = prior.uniform_prior(4, 6)
# We have a constraint on the extinction law (from Nogueras-Lara et al., 2019 here),
# so we can set Gaussian prior on the extinction law alpha
ext_alpha_prior = prior.gaussian_prior(2.23, 0.03)

param_priors = prior.prior_collection([
    star1_rad_prior,
    star2_rad_prior,
    bin_period_prior,
    bin_inc_prior,
    bin_t0_prior,
    bin_rv_com_prior,
    ext_f153m_prior,
    ext_alpha_prior,
])

param_names = [
    'star1_rad',
    'star2_rad',
    'bin_period',
    'bin_inc',
    'bin_t0',
    'bin_rv_com',
    'ext_f153m',
    'ext_alpha',
]

wrapped_params = [
    False,  # Star 1 Rad
    False,  # Star 2 Rad
    False,  # Binary period
    True,   # Binary inclination
    False,  # Binary t0
    False,  # Binary CoM RV
    False,  # F153M-band extinction
    False,  # Extinction law alpha
]

# Set up sampler and run
sampler = ultranest.ReactiveNestedSampler(
    param_names,
    loglike = un_evaluate,   # Function that we wrote for ultranest to calculate likelihood
    transform = param_priors.prior_transform_ultranest,  # Phitter's prior collection function to transform prior
    log_dir='./un_out/',
    resume='resume-similar',
    warmstart_max_tau=0.25,
)

# Step sampler (A "slice sampler" is often more efficient when fitting many parameters)
sampler.stepsampler = ultranest.stepsampler.SliceSampler(
    nsteps=(2**3)*len(param_names),
    generate_direction=ultranest.stepsampler.generate_mixture_random_direction,
)

result = sampler.run(
    show_status=True,
    update_interval_volume_fraction=0.98,
)

# Results runs in main process only
if MPI.COMM_WORLD.Get_rank() == 0:
    sampler.print_results()
    
    sampler.plot()
    
    sampler.plot_run()
    sampler.plot_trace()
    sampler.plot_corner()