import numpy as np
from scipy.integrate import trapezoid as simps
from scipy.signal import argrelmax

from ntrfc.math.methods import autocorr, zero_crossings


def integralscales(signal, timesteparray):
    mean = np.mean(signal)
    fluctations = signal - mean
    timesteps = timesteparray.copy()
    autocorrelated = autocorr(fluctations)
    # we are integrating from zero to zero-crossing in the autocorrelation, we need the time to begin with zeros
    # probably the used datasample is not beginning with 0. therefore:
    timesteps -= timesteps[0]
    if len(zero_crossings(autocorrelated)) > 0:
        acorr_zero_crossings = zero_crossings(autocorrelated)[0]
    else:
        print("no zero crossing found, using first minimal value (possibly last timestep). check data quality!")
        acorr_zero_crossings = np.where(autocorrelated == min(autocorrelated))[0][0]

    if all(np.isnan(autocorrelated)) or acorr_zero_crossings == 0:
        return 0
    integral_time_scale = simps(autocorrelated[:acorr_zero_crossings], timesteps[:acorr_zero_crossings])
    integral_length_scale = integral_time_scale * mean

    return integral_time_scale, integral_length_scale


def get_self_correlating_frequencies(timesteps, signal):
    """
    Calculate the self-correlating frequencies in a signal.

    Parameters
    ----------
    timesteps : ndarray
        An array of uniformly spaced timesteps, corresponding to the time intervals
        between samples in the signal.
    signal : ndarray
        The input signal to analyze.

    Returns
    -------
    frequencies : ndarray
        An array of self-correlating frequencies, in units of inverse timesteps.

    Notes
    -----
    This function calculates the autocorrelation of the input signal,
    and finds the local maxima in the autocorrelation signal.
    These local maxima correspond to the self-correlating frequencies in the signal.
    """
    # Calculate the autocorrelation of the signal
    autocorr_signal = autocorr(signal)

    # Find self-correlating frequencies by finding the local maxima in the autocorrelation signal
    self_correlating_frequencies = argrelmax(autocorr_signal)[0]
    ts_shifted = timesteps - timesteps[0]
    periods = ts_shifted[self_correlating_frequencies]
    frequencies = 1 / periods
    return frequencies
