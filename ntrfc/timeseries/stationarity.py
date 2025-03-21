import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy import signal
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress

from ntrfc.math.methods import minmax_normalize


def optimal_window_size(time_series, mean_error=0, min_interval=0.05, max_interval=0.25):
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
    allowed_window_sizes = range(min_window_size, max_window_size + 1)

    mean_scores = []
    std_scores = []
    for window_size in allowed_window_sizes:
        test_window = time_series[-2 * window_size:]
        test_rolling_window = pd.DataFrame(test_window).rolling(window=window_size)
        mean_scores.append(np.nanstd(test_rolling_window.mean().values))
        std_scores.append(np.nanstd(test_rolling_window.var().values))

    mean_scores = np.array(mean_scores)
    std_scores = np.array(std_scores)

    cumulated_scores = minmax_normalize(mean_scores) + minmax_normalize(std_scores)
    optimal_window_size_index = np.argmin(cumulated_scores)
    opt_window_size = allowed_window_sizes[optimal_window_size_index]

    reference_window = time_series[-2 * opt_window_size:]

    assert len(reference_window) == opt_window_size * 2

    if not all(reference_window == reference_window[0]):
        # check with a minmax normalized series and a rolling window.
        # needs to be normalized to have comparable values for a non-stationary check

        test_signal = minmax_normalize(reference_window)
        test_rolling_window = pd.DataFrame(test_signal).rolling(window=opt_window_size)
        test_mean = np.mean(test_signal)
        test_var = np.var(test_signal)

        test_mean_rolling = test_rolling_window.mean().values[:, 0]
        test_var_rolling = test_rolling_window.var().values[:, 0]

        test_mean_rolling_nonnan = test_mean_rolling[~np.isnan(test_mean_rolling)]
        test_var_rolling_nonnan = test_var_rolling[~np.isnan(test_var_rolling)]

        test_mean_rolling_error = test_mean_rolling_nonnan - test_mean
        test_var_rolling_error = test_var_rolling_nonnan - test_var

        allowed_mean_mean_error = 0.005 + mean_error  # test_mean_diff *0.3 + mean_error
        allowed_var_mean_error = 0.002 + mean_error  # test_var_diff*0.1 + mean_error

        error_mean = np.std(test_mean_rolling_error)
        error_var = np.std(test_var_rolling_error)

        if abs(error_mean) > allowed_mean_mean_error:
            print("Mean error too high")
            return False, False, False
        if error_var > allowed_var_mean_error:
            print("Variance error too high")
            return False, False, False

    return reference_window, opt_window_size, False


def last_coherent_interval(arr, opt_window_size, threshold=0.95):
    true_indices = np.where(arr == True)[0]
    last_false_index = true_indices[-1] + 1 if len(true_indices) > 0 else 0
    success_rate = np.array(
        [(np.sum(arr[:i]) + 2 * opt_window_size) / (len(arr[:i]) + 2 * opt_window_size) for i in range(len(arr))])
    success_rate[last_false_index:] = 0
    answer_index = np.where(success_rate >= threshold)[0][-1] + 1
    return answer_index


def estimate_stationarity(timeseries, allowed_error=0, create_plot_dir=False):
    if len(timeseries) < 400:
        return None, None, None

    sigma_threshold = 2
    percentage = (stats.norm.cdf(sigma_threshold) - stats.norm.cdf(-sigma_threshold))

    ref_window, opt_window_size, nperiods = optimal_window_size(timeseries, mean_error=allowed_error)
    if not opt_window_size:
        return None, None, None
    # normalize the reference window to have 0 mean and unit variance
    # if std is 0, then the window is constant, we cant devide by 0
    reference_window_normed = normalize_zscore(ref_window, np.std(ref_window), np.mean(ref_window))

    reference_mean = np.mean(reference_window_normed)
    reference_variance = np.var(reference_window_normed)

    opt_rolling_window = pd.DataFrame(reference_window_normed).rolling(window=opt_window_size)

    rolling_means = np.nan_to_num(opt_rolling_window.mean().values, 0)
    rolling_vars = np.nan_to_num(opt_rolling_window.var().values, 0)

    mean_uncertainty = np.nanstd(rolling_means)  # /np.sqrt(len(rolling_means))
    var_uncertainty = np.nanstd(rolling_vars)  # /np.sqrt(len(rolling_vars))

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

    mean_limits = (mean_uncertainty + mean_trend_tolerance) * 1.6 + allowed_error
    var_limits = (var_uncertainty + var_trend_tolerance) * 1.6 + allowed_error

    rolling_means_errors_inliers_reversed = rolling_means_errors_reversed <= mean_limits
    rolling_vars_errors_inliers_reversed = rolling_vars_errors_reversed <= var_limits

    mean_index = last_coherent_interval(rolling_means_errors_inliers_reversed, opt_window_size, percentage)

    variance_index = last_coherent_interval(rolling_vars_errors_inliers_reversed, opt_window_size, percentage)

    stationary_start_index = min(mean_index, variance_index)

    if create_plot_dir:
        plot_args = {"mean_index": mean_index,
                     "mean_limits": mean_limits,
                     "opt_window_size": opt_window_size,
                     "rolling_means_errors_inliers_reversed": rolling_means_errors_inliers_reversed,
                     "rolling_means_errors_reversed": rolling_means_errors_reversed,
                     "rolling_means_reversed": rolling_means_reversed,
                     "rolling_vars_errors_inliers_reversed": rolling_vars_errors_inliers_reversed,
                     "rolling_vars_errors_reversed": rolling_vars_errors_reversed,
                     "rolling_vars_reversed": rolling_vars_reversed,
                     "stationary_start_index": stationary_start_index,
                     "timeseries": timeseries,
                     "var_limits": var_limits,
                     "variance_index": variance_index}

        plot_stationarity_analysis(create_plot_dir, plot_args)

    return_index = len(timeseries) - (stationary_start_index)
    stationary_signal = timeseries[return_index:]
    stationary_rolling_signal = pd.DataFrame(stationary_signal).rolling(window=opt_window_size)
    mean_std = np.nanmax(stationary_rolling_signal.mean().values) - np.nanmin(stationary_rolling_signal.mean().values)

    return return_index, stationary_signal, mean_std


def plot_stationarity_analysis(plotdir, plot_args):
    mean_index = plot_args["mean_index"]
    mean_limits = plot_args["mean_limits"]
    opt_window_size = plot_args["opt_window_size"]
    rolling_means_errors_inliers_reversed = plot_args["rolling_means_errors_inliers_reversed"]
    rolling_means_errors_reversed = plot_args["rolling_means_errors_reversed"]
    rolling_means_reversed = plot_args["rolling_means_reversed"]
    rolling_vars_errors_inliers_reversed = plot_args["rolling_vars_errors_inliers_reversed"]
    rolling_vars_errors_reversed = plot_args["rolling_vars_errors_reversed"]
    rolling_vars_reversed = plot_args["rolling_vars_reversed"]
    stationary_start_index = plot_args["stationary_start_index"]
    timeseries = plot_args["timeseries"]
    var_limits = plot_args["var_limits"]
    variance_index = plot_args["variance_index"]

    _, axs = plt.subplots(4, 1, )
    axs[0].plot(rolling_vars_errors_inliers_reversed[::-1], label="var inliers")
    axs[0].axvspan(0, variance_index, facecolor='lightblue', alpha=0.5)
    axs[1].plot(np.array([(np.sum(rolling_vars_errors_inliers_reversed[:i]) + 2 * opt_window_size) / (
        len(rolling_vars_errors_inliers_reversed[:i]) + 2 * opt_window_size) for i in
                          range(len(rolling_vars_errors_inliers_reversed))])[::-1], label="var inliers percentage")
    axs[1].set_ylim(0.95, 1.01)
    axs[2].plot(rolling_vars_errors_reversed[::-1], label="var errors reversed")
    axs[2].axhline(var_limits, 0, len(rolling_vars_errors_reversed))
    axs[3].plot(rolling_vars_reversed[::-1], label="variance reversed")
    axs[3].set_ylabel("rolling dataframe id")
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[3].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plotdir, "variance_errors.png"))

    _, axs = plt.subplots(4, 1, )

    axs[0].plot(rolling_means_errors_inliers_reversed[::-1], label="mean inliers")
    axs[0].axvspan(len(timeseries) - mean_index, len(timeseries), facecolor='lightblue', alpha=0.5)
    axs[1].plot(np.array([(np.sum(rolling_means_errors_inliers_reversed[:i]) + 2 * opt_window_size) / (
        len(rolling_means_errors_inliers_reversed[:i]) + 2 * opt_window_size) for i in
                          range(len(rolling_means_errors_inliers_reversed))])[::-1], label="mean inliers percentage")
    axs[1].set_ylim(0.95, 1.01)
    axs[2].plot(rolling_means_errors_reversed[::-1], label="mean_errors_reversed")
    axs[2].axhline(mean_limits, 0, len(rolling_vars_errors_reversed))
    axs[3].plot(rolling_means_reversed[::-1], label="mean reversed")
    axs[3].set_ylabel("rolling dataframe id")
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[3].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plotdir, "mean_errors.png"))

    _, axs = plt.subplots(1, 1, )
    axs.plot(timeseries, label="timeseries")
    axs.axvline(len(timeseries) - stationary_start_index, label="stationarity start", color="red")
    axs.axvline(len(timeseries) - opt_window_size, min(timeseries), max(timeseries), label="rolling window size",
                color="grey")
    axs.axvline(len(timeseries), min(timeseries), max(timeseries), color="grey")
    axs.axvspan(len(timeseries) - opt_window_size * 2, len(timeseries), facecolor='lightgreen', alpha=0.5,
                label="reference window")
    axs.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plotdir, "timeseries.png"))


def normalize_minmax(ref_window):
    min_val = np.min(ref_window)
    max_val = np.max(ref_window)
    if min_val == max_val:
        reference_window_normed = (ref_window - min_val)
    else:
        reference_window_normed = (ref_window - min_val) / (max_val - min_val)
    return reference_window_normed


def normalize_zscore(ref_window, win_std, win_mean):
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

    mean_jk_error = np.std(mean_jk)
    var_jk_error = np.std(var_jk)

    return mean_jk_error, var_jk_error
