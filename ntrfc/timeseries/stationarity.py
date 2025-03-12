import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy import signal
from statsmodels.tsa.stattools import adfuller
from ntrfc.math.methods import minmax_normalize




def optimal_window_size(time_series, min_interval=0.05, max_interval=0.25, verbose=False):
    """
    Determines the optimal window size for a given time series.

    Parameters:
    ----------
    time_series : array-like
        The time series to analyze.
    verbose : bool, optional
        If True, a plot of the correlation coefficient and KS test results for each window size will be displayed.

    Returns:
    -------
    int or bool
        The optimal window size for the time series. If no suitable window size is found, False is returned.

    Notes:
    -----
    The function normalizes the input time series using the minmax_normalize() function.
    The window size is chosen based on a cumulative score that takes into account the correlation coefficient and
     KS test p-value.
    The function returns False if no suitable window size is found, meaning the input time series does not exhibit the
     necessary periodicity.
    """

    # Get the length of the time series and define a range of allowed window sizes
    series_length = len(time_series)
    min_window_size = int(series_length * min_interval)
    max_window_size = int(series_length * max_interval)
    allowed_window_sizes = range(min_window_size, max_window_size+1)

    mean_scores = []
    var_scores = []
    for window_size in allowed_window_sizes:
        test_window = time_series[-2 * window_size:]
        test_rolling_window = pd.DataFrame(test_window).rolling(window=window_size)
        mean_scores.append(np.nanstd(test_rolling_window.mean().values))
        var_scores.append(np.nanstd(test_rolling_window.var().values))


    mean_scores = np.array(mean_scores)
    var_scores = np.array(var_scores)

    cumulated_scores = minmax_normalize(mean_scores) + minmax_normalize(var_scores)
    optimal_window_size_index = np.argmin(cumulated_scores)
    opt_window_size = allowed_window_sizes[optimal_window_size_index]

    reference_window = time_series[-2 * opt_window_size:]

    assert len(reference_window) == opt_window_size * 2

    if not all(reference_window == reference_window[0]):
        # check with a minmax normalized series and a rolling window.
        # needs to be normalized to have comparable values for a non-stationary check
        test_rolling_window = pd.DataFrame(minmax_normalize(reference_window)).rolling(window=opt_window_size)
        meandiff = (test_rolling_window.mean().max()-test_rolling_window.mean().min()).iloc[0]
        vardiff = (test_rolling_window.var().max()-test_rolling_window.var().min()).iloc[0]
        # is non-stationary if the mean and variance are not constant
        if meandiff> 1/30 or vardiff >1/30 :
            return False, False, False

    return reference_window, opt_window_size, False


def last_coherent_interval(arr, opt_window_size,threshold=0.95):
    true_indices = np.where(arr == True)[0]
    last_false_index = true_indices[-1]+1 if len(true_indices) > 0 else 0
    success_rate = np.array([(np.sum(arr[:i])+2*opt_window_size) / (len(arr[:i])+2*opt_window_size)  for i in range(len(arr))])
    success_rate[last_false_index:] = 0
    answer_index = np.where(success_rate >= threshold)[0][-1]+1
    return answer_index


def estimate_stationarity(timeseries):
    sigma_threshold = 3
    percentage = (stats.norm.cdf(sigma_threshold) - stats.norm.cdf(-sigma_threshold))

    ref_window, opt_window_size, nperiods = optimal_window_size(timeseries)
    if not opt_window_size:
        return False
    # normalize the reference window to have 0 mean and unit variance
    # if std is 0, then the window is constant, we cant devide by 0
    reference_window_normed = normalize_zscore(ref_window, np.std(ref_window), np.mean(ref_window))

    reference_mean = np.mean(reference_window_normed)
    reference_variance = np.var(reference_window_normed)
    opt_rolling_window = pd.DataFrame(reference_window_normed).rolling(window=opt_window_size)

    rolling_means = np.nan_to_num(opt_rolling_window.mean().values,0)
    rolling_vars = np.nan_to_num(opt_rolling_window.var().values,0)

    mean_uncertainty = np.nanstd(rolling_means)#/np.sqrt(len(rolling_means))
    var_uncertainty = np.nanstd(rolling_vars)#/np.sqrt(len(rolling_vars))

    mean_trend_tolerance = (np.nanmax(rolling_means) - np.nanmin(rolling_means))
    var_trend_tolerance = (np.nanmax(rolling_vars) - np.nanmin(rolling_vars))

    checkseries = normalize_zscore(timeseries, np.std(ref_window), np.mean(ref_window))

    checkseries_reversed = pd.DataFrame(checkseries[::-1])
    rolling_win_reversed = checkseries_reversed.rolling(window=opt_window_size)

    rolling_means_reversed = rolling_win_reversed.mean().values
    rolling_vars_reversed = rolling_win_reversed.var().values

    rolling_means_reversed = np.nan_to_num(rolling_means_reversed, nan=reference_mean)
    rolling_vars_reversed = np.nan_to_num(rolling_vars_reversed, nan=reference_variance)

    rolling_means_errors_reversed = np.abs(rolling_means_reversed - reference_mean)
    rolling_vars_errors_reversed = np.abs(rolling_vars_reversed - reference_variance)

    mean_limits = (mean_uncertainty + mean_trend_tolerance) * (sigma_threshold)
    var_limits = (var_uncertainty + var_trend_tolerance) * (sigma_threshold)

    rolling_means_errors_inliers_reversed = rolling_means_errors_reversed  <= mean_limits
    rolling_vars_errors_inliers_reversed = rolling_vars_errors_reversed  <= var_limits


    mean_index = last_coherent_interval(rolling_means_errors_inliers_reversed,opt_window_size,percentage)

    variance_index = last_coherent_interval(rolling_vars_errors_inliers_reversed,opt_window_size,percentage)

    stationary_start_index = min(mean_index, variance_index)

    # import matplotlib.pyplot as plt
    #
    # plt.figure(figsize=(120, 120))
    # _, axs = plt.subplots(9, 1, )
    # axs[0].plot(rolling_vars_errors_inliers_reversed, label="var inliers")
    # axs[0].axvspan(0, variance_index, facecolor='lightblue', alpha=0.5)
    # axs[1].plot(np.array([(np.sum(rolling_vars_errors_inliers_reversed[:i]) + 2 * opt_window_size) / (
    #     len(rolling_vars_errors_inliers_reversed[:i]) + 2 * opt_window_size) for i in
    #                       range(len(rolling_vars_errors_inliers_reversed))]))
    # axs[1].set_ylim(0.95, 1)
    # axs[2].plot(rolling_vars_errors_reversed, label="var errors reversed")
    # axs[2].axhline(var_limits, 0, len(rolling_vars_errors_reversed))
    # axs[3].plot(rolling_vars_reversed, label="variance reversed")
    # axs[4].plot(rolling_means_errors_inliers_reversed, label="mean inliers")
    # axs[4].axvspan(0, mean_index, facecolor='lightblue', alpha=0.5)
    # axs[5].plot(np.array([(np.sum(rolling_means_errors_inliers_reversed[:i]) + 2 * opt_window_size) / (
    #     len(rolling_means_errors_inliers_reversed[:i]) + 2 * opt_window_size) for i in
    #                       range(len(rolling_means_errors_inliers_reversed))]))
    # axs[5].set_ylim(0.95, 1)
    # axs[6].plot(rolling_means_errors_reversed, label="mean_errors_reversed")
    # axs[6].axhline(mean_limits, 0, len(rolling_vars_errors_reversed))
    # axs[7].plot(rolling_means_reversed, label="mean reversed")
    # axs[8].plot(timeseries[::-1], label="timeseries")
    # axs[8].axvline(stationary_start_index + 2 * opt_window_size)
    # axs[8].axvspan(0, opt_window_size * 2, facecolor='lightblue', alpha=0.5)
    # axs[8].axvspan(opt_window_size * 2, opt_window_size * 3, facecolor='lightgreen', alpha=0.5)
    # axs[0].legend()
    # axs[1].legend()
    # axs[2].legend()
    # axs[3].legend()
    # axs[4].legend()
    # axs[5].legend()
    # axs[6].legend()
    # axs[7].legend()
    # axs[8].legend()
    # # plt.tight_layout()
    # plt.show()
    return_index = len(timeseries)-(stationary_start_index)
    return return_index


def normalize_minmax(ref_window):
    min_val = np.min(ref_window)
    max_val = np.max(ref_window)
    if min_val == max_val:
        reference_window_normed = (ref_window - min_val)
    else:
        reference_window_normed = (ref_window - min_val) / (max_val - min_val)
    return reference_window_normed

def normalize_zscore(ref_window,win_std,win_mean):
    if win_std == 0:
        reference_window_normed = (ref_window - win_mean)
    else:
        reference_window_normed = (ref_window - win_mean) / win_std
    return reference_window_normed


def estimate_error_jacknife(timeseries, block_size=20, n_samples=4000):
    """
    Estimates the errors of the mean, variance, and autocorrelation of a given time series using jackknife resampling method.

    Parameters
    ----------
    timeseries : array-like
        The input time series.
    block_size : int, optional
        The block size used in the jackknife resampling method (default is 20).
    n_samples : int, optional
        The number of jackknife samples to generate (default is 4000).

    Returns
    -------
    tuple
        A tuple of three floats representing the error estimates of the mean, variance, and autocorrelation, respectively.

    Notes
    -----
    The jackknife resampling method is used to estimate the errors of the mean, variance, and autocorrelation of the
     input time series.
    The function generates `n_samples` jackknife samples by randomly selecting blocks from the time series calculates
     the mean, variance, and autocorrelation of each jackknife sample.
    It also generates `n_samples` noise samples with the same block size as the original time series and calculates the
     mean, variance, and autocorrelation of each noise sample.
    The standard deviation of the jackknife estimates for mean, variance, and autocorrelation are calculated, and each
     is multiplied by a factor of 16 to obtain the final error estimates.

    Choosing an appropriate block size is crucial for obtaining reliable and accurate estimates of the errors of the
     mean, variance, and autocorrelation of a given time series using the jackknife resampling method.
    """

    # Original time series

    x = timeseries
    #
    n_blocks = len(timeseries) // block_size

    # Initialize arrays to store jackknife estimates
    mean_jk = np.zeros(n_samples)
    var_jk = np.zeros(n_samples)
    acorr_jk = np.zeros(n_samples)

    for i in range(n_samples):
        # Generate a random index array of block indices
        idx = np.random.randint(0, n_blocks, size=n_blocks)
        # Select blocks according to the random indices
        start = idx * block_size
        end = start + block_size
        x_jk = np.concatenate([x[start[i]:end[i]] for i in range(len(start))])

        # Calculate the mean, variance, and autocorrelation of the jackknife sample
        mean_jk[i] = np.mean(x_jk)
        var_jk[i] = np.var(x_jk)

        freqs, psd = signal.periodogram(x_jk, return_onesided=False)
        acorr_jk[i] = freqs[np.argmax(psd)]

    mean_jk_error = np.std(mean_jk)
    var_jk_error = np.std(var_jk)
    accr_jk_error = np.std(acorr_jk)

    return mean_jk_error, var_jk_error, accr_jk_error
