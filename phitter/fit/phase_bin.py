# Class to assist in binning observations by phase
# ---
# Abhimat Gautam

import numpy as np
from phitter import observables

class phase_bin(object):
    """
    `phase_bin` is a class to assist in binning observations by phase.
    Instantiate this object with an observables object to indicate the
    observed observables.
    
    Parameters
    ----------
    observations : observables
        observables object, specified with the observations to be fit.
    """
    
    def __init__(self, observations):
        # Save observations to the object
        self.observations = observations
    
    def phase_in_bins(
        self, phase_period, phase_t0,
        num_bins=100, normalize_uncs=False,
    ):
        """
        Function to perform the phasing of observations.
        
        Parameters
        ----------
        phase_period : float
            Period to phase the observations to.
        phase_t0 : float
            t0 to use when performing the phasing.
        num_bins : int, default=100
            Number of bins to use when phasing. Makes `num_bins` equal sized
            bins between phase of 0 -- 1.
        normalized_uncs : boolean, default=False
            Whether or not to normalize the output uncertainties. If False,
            output uncertainties reflect number of data points in each phase
            bin. If True, output uncertainties for each phase bin are normalized
            so that more observations do not result in smaller uncertainties for
            a given bin.
        
        Returns
        -------
        phased_observables : observables
            A substitute observables object
        phased_model_observables : observables
        """
        
        bin_spacing = 1./num_bins
        
        bin_starts = np.arange(0, 1, bin_spacing)
        bin_ends = np.arange(bin_spacing, 1+bin_spacing, bin_spacing)
        
        out_obs_times = np.array([])
        out_obs = np.array([])
        out_obs_uncs = np.array([])
        out_obs_filts = np.array([])
        out_obs_types = np.array([])
        
        # Phase the photometric data
        if self.observations.num_filts_phot > 0:
            # Empty arrays for output of all binned photometric data
            binned_obs_times = np.full(
                self.observations.num_filts_phot * num_bins, np.nan,
            )
            binned_obs = np.full(
                self.observations.num_filts_phot * num_bins, np.nan,
            )
            binned_obs_uncs = np.full(
                self.observations.num_filts_phot * num_bins, np.nan,
            )
            binned_obs_filts = np.array([])
            
            # Go through each photometric data filter
            for filt_index, filt in enumerate(self.observations.unique_filts_phot):
                # Cut out obs for current filter
                cur_filt_filter = self.observations.phot_filt_filters[filt]
                
                cur_filt_obs_times = self.observations.obs_times[cur_filt_filter]
                cur_filt_obs = self.observations.obs[cur_filt_filter]
                cur_filt_obs_uncs = self.observations.obs_uncs[cur_filt_filter]
                
                cur_filt_phases = ((cur_filt_obs_times - phase_t0) % phase_period) / phase_period
                
                # Empty arrays for gathering binned data
                filt_binned_obs = np.full(num_bins, np.nan)
                filt_binned_obs_uncs = np.full(num_bins, np.nan)
                filt_binned_phases = np.full(num_bins, np.nan)
                
                # Go through each phase bin
                for bin_index in range(num_bins):
                    bin_start = bin_starts[bin_index]
                    bin_end = bin_ends[bin_index]
                    
                    phase_bin_filt = np.where(np.logical_and(
                        cur_filt_phases >= bin_start,
                        cur_filt_phases < bin_end,
                    ))
                    
                    num_bin_points = len(phase_bin_filt[0])
                    
                    # Go to next bin if no observations in current bin
                    if num_bin_points == 0:
                        continue
                    
                    bin_obs = cur_filt_obs[phase_bin_filt]
                    bin_obs_uncs = cur_filt_obs_uncs[phase_bin_filt]
                    bin_phases = cur_filt_phases[phase_bin_filt]
                    
                    # Weights for weighted mean
                    bin_weights = 1. / (bin_obs_uncs ** 2.)
                    
                    # Calculate weighted mean observations and uncertainties
                    binned_mag = np.average(bin_obs, weights=bin_weights)
                    binned_obs_unc = np.sqrt(1./(np.sum(bin_obs_uncs**-2)))
                    binned_phase = np.average(bin_phases, weights=bin_weights)
                    
                    if normalize_uncs:
                        binned_obs_unc = (num_bin_points**0.5) * binned_obs_unc
                    
                    filt_binned_obs[bin_index] = binned_mag
                    filt_binned_obs_uncs[bin_index] = binned_obs_unc
                    filt_binned_phases[bin_index] = binned_phase
                    
                # Store current filter's binned values in output
                filt_slice = slice(filt_index*num_bins, (filt_index+1)*num_bins)
                
                binned_obs_times[filt_slice] = (filt_binned_phases * phase_period) + phase_t0
                binned_obs[filt_slice] = filt_binned_obs
                binned_obs_uncs[filt_slice] = filt_binned_obs_uncs
                
                # Make filts array
                binned_obs_filts = np.append(binned_obs_filts, np.full(
                    num_bins, filt
                ))
            
            # Clear out nans
            empty_cut_filter = np.where(np.isfinite(binned_obs))
            
            binned_obs_times = binned_obs_times[empty_cut_filter]
            binned_obs = binned_obs[empty_cut_filter]
            binned_obs_uncs = binned_obs_uncs[empty_cut_filter]
            binned_obs_filts = binned_obs_filts[empty_cut_filter]
            
            binned_obs_types = np.full(
                len(binned_obs_times), 'phot',
            )
            
            out_obs_times = np.append(out_obs_times, binned_obs_times)
            out_obs = np.append(out_obs, binned_obs)
            out_obs_uncs = np.append(out_obs_uncs, binned_obs_uncs)
            out_obs_filts = np.append(out_obs_filts, binned_obs_filts)
            out_obs_types = np.append(out_obs_types, binned_obs_types)
        
        # Phase the RV data
        if self.observations.num_filts_rv > 0:
            # Empty arrays for output of all binned RV data
            binned_obs_times = np.full(
                self.observations.num_filts_rv * num_bins * 2, np.nan,
            )
            binned_obs = np.full(
                self.observations.num_filts_rv * num_bins * 2, np.nan,
            )
            binned_obs_uncs = np.full(
                self.observations.num_filts_rv * num_bins * 2, np.nan,
            )
            binned_obs_filts = np.array([])
            binned_obs_types = np.full(
                self.observations.num_filts_rv * num_bins * 2, 'rv_pri',
            )
            
            # Go through each RV data filter for both stars
            for filt_index, filt in enumerate(self.observations.unique_filts_rv):
                # First carry out for primary star
                
                # Cut out obs for current filter
                cur_filt_filter = np.logical_and(
                    self.observations.obs_filts == filt,
                    self.observations.obs_types == 'rv_pri',
                )
                
                cur_filt_obs_times = self.observations.obs_times[cur_filt_filter]
                cur_filt_obs = self.observations.obs[cur_filt_filter]
                cur_filt_obs_uncs = self.observations.obs_uncs[cur_filt_filter]
                
                cur_filt_phases = ((cur_filt_obs_times - phase_t0) % phase_period) / phase_period
                
                # Empty arrays for gathering binned data
                filt_binned_obs = np.full(num_bins, np.nan)
                filt_binned_obs_uncs = np.full(num_bins, np.nan)
                filt_binned_phases = np.full(num_bins, np.nan)
                
                # Go through each phase bin
                for bin_index in range(num_bins):
                    bin_start = bin_starts[bin_index]
                    bin_end = bin_ends[bin_index]
                    
                    phase_bin_filt = np.where(np.logical_and(
                        cur_filt_phases >= bin_start,
                        cur_filt_phases < bin_end,
                    ))
                    
                    num_bin_points = len(phase_bin_filt[0])
                    
                    # Go to next bin if no observations in current bin
                    if num_bin_points == 0:
                        continue
                    
                    bin_obs = cur_filt_obs[phase_bin_filt]
                    bin_obs_uncs = cur_filt_obs_uncs[phase_bin_filt]
                    bin_phases = cur_filt_phases[phase_bin_filt]
                    
                    # Weights for weighted mean
                    bin_weights = 1. / (bin_obs_uncs ** 2.)
                    
                    # Calculate weighted mean observations and uncertainties
                    binned_rv = np.average(bin_obs, weights=bin_weights)
                    binned_obs_unc = np.sqrt(1./(np.sum(bin_obs_uncs**-2)))
                    binned_phase = np.average(bin_phases, weights=bin_weights)
                    
                    if normalize_uncs:
                        binned_obs_unc = (num_bin_points**0.5) * binned_obs_unc
                    
                    filt_binned_obs[bin_index] = binned_rv
                    filt_binned_obs_uncs[bin_index] = binned_obs_unc
                    filt_binned_phases[bin_index] = binned_phase
                    
                # Store current filter's binned values in output
                filt_slice = slice(filt_index*num_bins, (filt_index+1)*num_bins)
                
                binned_obs_times[filt_slice] = (filt_binned_phases * phase_period) + phase_t0
                binned_obs[filt_slice] = filt_binned_obs
                binned_obs_types[filt_slice] = filt_binned_obs_uncs
                binned_obs_types[filt_slice] = np.full(
                    num_bins, 'rv_pri',
                )
                
                # Make filts array
                binned_obs_filts = np.append(binned_obs_filts, np.full(
                    num_bins, filt
                ))
                
                # Repeat for secondary star
                
                # Cut out obs for current filter
                cur_filt_filter = np.logical_and(
                    self.observations.obs_filts == filt,
                    self.observations.obs_types == 'rv_sec',
                )
                
                cur_filt_obs_times = self.observations.obs_times[cur_filt_filter]
                cur_filt_obs = self.observations.obs[cur_filt_filter]
                cur_filt_obs_uncs = self.observations.obs_uncs[cur_filt_filter]
                
                cur_filt_phases = ((cur_filt_obs_times - phase_t0) % phase_period) / phase_period
                
                # Empty arrays for gathering binned data
                filt_binned_obs = np.full(num_bins, np.nan)
                filt_binned_obs_uncs = np.full(num_bins, np.nan)
                filt_binned_phases = np.full(num_bins, np.nan)
                
                # Go through each phase bin
                for bin_index in range(num_bins):
                    bin_start = bin_starts[bin_index]
                    bin_end = bin_ends[bin_index]
                    
                    phase_bin_filt = np.where(np.logical_and(
                        cur_filt_phases >= bin_start,
                        cur_filt_phases < bin_end,
                    ))
                    
                    num_bin_points = len(phase_bin_filt[0])
                    
                    # Go to next bin if no observations in current bin
                    if num_bin_points == 0:
                        continue
                    
                    bin_obs = cur_filt_obs[phase_bin_filt]
                    bin_obs_uncs = cur_filt_obs_uncs[phase_bin_filt]
                    bin_phases = cur_filt_phases[phase_bin_filt]
                    
                    # Weights for weighted mean
                    bin_weights = 1. / (bin_obs_uncs ** 2.)
                    
                    # Calculate weighted mean observations and uncertainties
                    binned_rv = np.average(bin_obs, weights=bin_weights)
                    binned_obs_unc = np.sqrt(1./(np.sum(bin_obs_uncs**-2)))
                    binned_phase = np.average(bin_phases, weights=bin_weights)
                    
                    if normalize_uncs:
                        binned_obs_unc = (num_bin_points**0.5) * binned_obs_unc
                    
                    filt_binned_obs[bin_index] = binned_rv
                    filt_binned_obs_uncs[bin_index] = binned_obs_unc
                    filt_binned_phases[bin_index] = binned_phase
                    
                # Store current filter's binned values in output
                filt_slice = slice(
                    self.observations.num_filts_rv + filt_index*num_bins,
                    self.observations.num_filts_rv + (filt_index+1)*num_bins,
                )
                
                binned_obs_times[filt_slice] = (filt_binned_phases * phase_period) + phase_t0
                binned_obs[filt_slice] = filt_binned_obs
                binned_obs_types[filt_slice] = filt_binned_obs_uncs
                binned_obs_types[filt_slice] = np.full(
                    num_bins, 'rv_sec',
                )
                
                # Make filts array
                binned_obs_filts = np.append(binned_obs_filts, np.full(
                    num_bins, filt
                ))
            
            # Clear out nans
            empty_cut_filter = np.where(np.isfinite(binned_obs))
            
            binned_obs_times = binned_obs_times[empty_cut_filter]
            binned_obs = binned_obs[empty_cut_filter]
            binned_obs_uncs = binned_obs_uncs[empty_cut_filter]
            binned_obs_filts = binned_obs_filts[empty_cut_filter]
            binned_obs_types = binned_obs_types[empty_cut_filter]
            
            out_obs_times = np.append(out_obs_times, binned_obs_times)
            out_obs = np.append(out_obs, binned_obs)
            out_obs_uncs = np.append(out_obs_uncs, binned_obs_uncs)
            out_obs_filts = np.append(out_obs_filts, binned_obs_filts)
            out_obs_types = np.append(out_obs_types, binned_obs_types)
        
        # Create new observables object with phased observations
        new_binned_observations = observables.observables(
            obs_times=out_obs_times, obs=out_obs, obs_uncs=out_obs_uncs,
            obs_filts=out_obs_filts, obs_types=out_obs_types,
        )
        
        return new_binned_observations
        