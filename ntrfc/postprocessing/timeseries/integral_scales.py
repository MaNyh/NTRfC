import numpy as np
from scipy.integrate import simps

from ntrfc.utils.math.methods import autocorr, zero_crossings


def integralscales(signal, timesteparray):
    mean = np.mean(signal)
    fluctations = signal-mean
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

    if all(np.isnan(autocorrelated)) or acorr_zero_crossings==0:
        return 0, 0
    integral_time_scale = simps(autocorrelated[:acorr_zero_crossings], timesteps[:acorr_zero_crossings])
    integral_length_scale = integral_time_scale * mean

    return integral_time_scale, integral_length_scale
