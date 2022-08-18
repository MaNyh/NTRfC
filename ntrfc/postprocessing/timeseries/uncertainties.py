import numpy as np
import matplotlib.pyplot as plt
import bisect
from scipy.integrate import simps


def stationarity_uncertainties(timesteps, values, verbose=False):
    '''
    :param timesteps: 1D np.array()
    :param values: 1D np.array()
    :param verbose: bool
    :return: stationary_timestep ,mean_value , uncertainty
    '''

    ### commented is for rms

    # confidence_mean_high = [];  # upper limit
    # confidence_mean_low = [];  # lower limit
    # mean_in_ci_range = [];  # is value in confidence range?
    # out_of_range = [];

    # we are integrating from zero to zero-crossing in the autocorrelation, we need the time to begin with zeros
    # probably the used datasample is not beginning with 0. therefore:
    timesteps -= timesteps[0]  ###### revDA: timeSteps start at zero before signal is divided into two parts
    #################################################

    # this is for the full signal (technically not needed)
    # signal_mean = np.mean(values)

    # this is for half the signal
    second_half_id = int(len(values) / 2)
    second_half_of_signal = np.copy(values[second_half_id:])
    second_half_mean = np.mean(second_half_of_signal)
    second_half_of_signal_norm = second_half_of_signal - second_half_mean  ## the autocorrelation function needs a signal that fluctuates around 0
    autocorrelated = autocorr(second_half_of_signal_norm)
    second_half_timesteps = np.copy(timesteps[second_half_id:])

    if len(zero_crossings(autocorrelated)) > 0:
        acorr_zero_crossings = zero_crossings(autocorrelated)[0]
    else:
        print("no zero crossing found, using first minimal value (possibly last timestep). check data quality!")
        acorr_zero_crossings = np.where(autocorrelated == min(autocorrelated))[0][0]

    integral_scale = simps(autocorrelated[:acorr_zero_crossings], timesteps[:acorr_zero_crossings])

    # calculate the moving average for every 30*integral_scale time intervals
    integrals_window = 30
    t_window_size = integrals_window * integral_scale
    t_window_len = bisect.bisect(timesteps, t_window_size)
    # calculate moving average with t_window_len
    signal_SMA = moving_average(values, t_window_len)

    eps_time_mean = (np.std(second_half_of_signal) / second_half_mean) * (
        2 * integral_scale / (second_half_timesteps[-1] - second_half_timesteps[0])) ** .5
    # eps_time_rms = (integral_scale / (second_half_timesteps[-1] - second_half_timesteps[0])) ** .5

    n_windows_mean = len(signal_SMA)
    # n_windows_rms = len(windows_rms)

    confidence_level = 1.96
    confidence_mean_high = second_half_mean * (1 + confidence_level * eps_time_mean)
    confidence_mean_low = second_half_mean * (1 - confidence_level * eps_time_mean)
    # confidence_rms_high = np.mean(windows_rms) * (1 + 1.96 * eps_time_rms)
    # confidence_rms_low = np.mean(windows_rms) * (1 - 1.96 * eps_time_rms)

    mean_in_ci_range = [True if (confidence_mean_high >= i >= confidence_mean_low) else False for i in signal_SMA]
    # rms_in_ci_range = [True if (confidence_rms_high >= i >= confidence_rms_low) else False for i in windows_rms]

    mean_stationarity_fraction = np.array(
        [sum(mean_in_ci_range[pt:]) / (n_windows_mean - pt) for pt in range(len(mean_in_ci_range))])
    # rms_stationarity_fraction = np.array(
    #     [sum(rms_in_ci_range[pt:]) / (n_windows_rms - pt) for pt in range(len(rms_in_ci_range))])

    confidence_threshold = 0.95
    id_stationary = bisect.bisect(mean_stationarity_fraction, confidence_threshold)

    if verbose==True:
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(values)
        axs[0].hlines(confidence_mean_high, xmin=0, xmax=len(values), colors="cyan")
        axs[0].hlines(confidence_mean_low, xmin=0, xmax=len(values), colors="cyan")
        axs[0].plot(signal_SMA, "k", linewidth=.5)
        axs[0].vlines(id_stationary, ymin=min(values), ymax=max(values), colors="red")
        axs[1].plot(autocorrelated)
        plt.show()

    return timesteps(id_stationary), (confidence_mean_high-second_half_mean)



def zero_crossings(data_series):
    zcs = np.where(np.diff(np.sign(data_series)))[0]
    return zcs


def autocorr(x):
    norm = np.sum(np.array(x) ** 2)
    result = np.correlate(np.array(x), np.array(x), 'full') / norm
    return result[int(len(result) / 2):]


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w
