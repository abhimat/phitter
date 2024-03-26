# Functions to perform adjustments to photometry
# ---
# Abhimat Gautam

import numpy as np
from phoebe import u
from spisea import reddening

red_law_funcs_ks = {
    'NL18': reddening.RedLawNoguerasLara18().NoguerasLara18,
    'F11': reddening.RedLawFritz11().Fritz11,
}


def apply_distance_modulus(bin_observables, target_dist):
    """
    Modeled observables are calculated at a distance of 10 pc. Adds distance
    modulus to modeled fluxes in observables object.
    
    Parameters
    ----------
    bin_observables : observables
        observables object containing photometry to which to apply distance
        modulus.
    target_dist : Astropy Quantity, length unit
        Distance to modeled binary.
    
    Returns
    -------
    observables
        observables object, where photometry has distance modulus added.
    """
    # Return if there are no photometry observations
    if bin_observables.num_obs_phot == 0:
        return bin_observables
    
    # Calculate distance modulus
    dist_modulus = 5. * np.log10(target_dist / (10. * u.pc))
    
    bin_observables.obs[bin_observables.obs_phot_filter] += dist_modulus
    
    return bin_observables

    
def apply_extinction(
        bin_observables, isoc_Ks_ext,
        ref_filt, target_ref_filt_ext,
        isoc_red_law='NL18', ext_alpha=None,
    ):
    """
    Modeled observables are calculated without extinction. Add extinction /
    reddening to modeled fluxes in observables object.
    
    Parameters
    ----------
    bin_observables : observables
        observables object containing photometry to which to apply distance
        modulus.
    isoc_Ks_ext : float
        Extinction, in Ks band, for the isochrone object used to generate
        stellar parameters.
    ref_filt : filter
        filter object, corresponding to the reference filter / passband
        from which all other extinctions are calculated.
    target_ref_filt_ext : float
        Extinction in the ref_filt, A_{ref_filt}, from which extinction in other
        filters is calculated using a power law extinction law.
    ext_alpha : float or None, default=None
        If specified, the power law slope for an extinction law from which the
        extinction at filters other than the ref_filt are calculated usin the
        extinction in the ref_filt (target_ref_filt_ext). If none,
        the extinction in other filters is just applied using the extinction
        implied by the isochrone object's Ks-band extinction.
    
    Returns
    -------
    observables
        observables object, where photometry has distance modulus added.
    """
    # Return if there are no photometry observations
    if bin_observables.num_obs_phot == 0:
        return bin_observables
    
    # Determine isochrone extinctions and extinction adjustments for each filter
    isoc_filts_ext = {}
    filts_ext_adj = {}
    
    for cur_filt in bin_observables.unique_filts_phot:
        # Determine extinction in current band from the extinction law used and
        # the reference extinction from the isochrone object
        isoc_filts_ext[cur_filt] = red_law_funcs_ks[isoc_red_law](
            cur_filt.lambda_filt.to(u.micron).value,
            isoc_Ks_ext,
        )[0]
        
        # Allow a tweaked to the extinction law, with a tweaked power law slope.
        # Currently implemented as an extinction implied by a new power law
        # slope.
        if ext_alpha is not None:
            filts_ext_adj[cur_filt] = (\
                ((target_ref_filt_ext *
                    (ref_filt.lambda_filt / cur_filt.lambda_filt)**ext_alpha)
                 - isoc_filts_ext[cur_filt])
            )
        
        obs_filt_filter = np.where(np.logical_and(
            bin_observables.obs_types == 'phot',
            bin_observables.obs_filts == cur_filt,
        ))
        
        bin_observables.obs[obs_filt_filter] += \
            isoc_filts_ext[cur_filt] + filts_ext_adj[cur_filt]
        
    return bin_observables
