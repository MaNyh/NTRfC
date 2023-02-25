import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.stats import ks_2samp
from statsmodels.tsa.stattools import adfuller

from ntrfc.math.methods import minmax_normalize


def optimal_window_size(time_series, verbose=False):
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

    # Normalize the time series
    normalized_series = minmax_normalize(time_series)

    # Initialize lists to store correlation coefficient, KS test p-value, and sanity test p-value
    correlation_coefficients = []
    ks_p_values = []
    sanity_p_values = []

    # Get the length of the time series and define a range of allowed window sizes
    series_length = len(normalized_series)
    allowed_window_sizes = np.array(range(series_length // 20, series_length // 4))

    # Define a threshold for the sanity test
    sanity_threshold = 0.95

    # Iterate through the allowed window sizes and perform the Kolmogorov-Smirnov test and correlation coefficient calculation
    for window_size in allowed_window_sizes:
        check_window = normalized_series[series_length - (window_size * 2):]
        first_subsequence = check_window[:len(check_window) - window_size]
        second_subsequence = check_window[len(check_window) - window_size:]
        ks_statistic, ks_p_value = ks_2samp(second_subsequence, first_subsequence, alternative="less", mode='exact')
        if not np.array_equal(first_subsequence, first_subsequence):
            correlation_coefficients.append((1 + np.corrcoef(first_subsequence, second_subsequence)[0][1]) / 2)
        else:
            correlation_coefficients.append(1)
        ks_p_values.append(1 - ks_statistic)
        sanity_p_values.append(ks_p_value)

    # Calculate a cumulative score for each window size based on the correlation coefficient and KS test p-value
    corr_coeff_scores = np.nan_to_num(correlation_coefficients, 0)
    ks_pvalue_scores = np.array(ks_p_values)
    cumulated_scores = corr_coeff_scores + ks_pvalue_scores

    # Filter out window sizes with low sanity test p-values
    invalid_indices = np.where(np.array(sanity_p_values) < sanity_threshold)[0]
    cumulated_scores[invalid_indices] = 0

    max_indices = np.where(2 * sanity_threshold < cumulated_scores)[0]
    if len(max_indices) == 0:
        return False
    max_scores = cumulated_scores[max_indices]
    max_index = np.argmax(max_scores)
    optimal_window_size_index = max_indices[max_index]
    optimal_window_size = allowed_window_sizes[optimal_window_size_index]

    # If verbose mode is enabled, display a plot of the correlation coefficient and KS test results for each window size
    if verbose:
        plt.plot(np.nan_to_num(correlation_coefficients, 0), label="correlation coefficient")
        plt.plot(ks_p_values, label="KS test p-value")
        plt.axvline(optimal_window_size_index)
        plt.legend()
        plt.show()

    return optimal_window_size


def estimate_stationarity(timeseries, verbose=False):
    timeseries = minmax_normalize(timeseries)
    series_length = len(timeseries)
    reference_window_size = optimal_window_size(timeseries) * 2
    if not reference_window_size:
        return False

    reference_window = timeseries[series_length - reference_window_size:]
    reference_freqs, reference_psd = signal.periodogram(reference_window, return_onesided=False)

    reference_mean = np.mean(reference_window)
    reference_autocorr_frequency = reference_freqs[np.argmax(reference_psd)]
    reference_variance = np.var(reference_window)

    reference_autocorr_timescale = (
        1 / reference_autocorr_frequency) if reference_autocorr_frequency != 0 else reference_window_size // 2
    reference_nperiods = np.ceil(reference_window_size / reference_autocorr_timescale)
    block_size = int(reference_window_size / (reference_nperiods))
    mean_error, variance_error, autocorr_error = estimate_error(reference_window, block_size=block_size)

    rolling_window = np.lib.stride_tricks.sliding_window_view(timeseries, reference_window_size)

    window_means = np.mean(rolling_window, axis=1)
    mean_valid = np.isclose(abs(window_means - reference_mean), 0, atol=mean_error)
    mean_scores = np.cumsum(mean_valid[::-1])[::-1] / np.arange(1, len(mean_valid) + 1)[::-1]
    max_confi = 1

    window_corrs = []
    for ts in rolling_window:
        freqs, psd = signal.periodogram(ts, return_onesided=False)
        window_corrs.append(freqs[np.argmax(psd)])
    autocorr_timescale_valid = np.isclose(abs(window_corrs - reference_autocorr_frequency), 0,
                                          atol=autocorr_error)

    autocorr_timescale_scores = np.cumsum(autocorr_timescale_valid[::-1])[::-1] / np.arange(1,
                                                                                            len(autocorr_timescale_valid) + 1)[
                                                                                  ::-1]

    autocorr_timescale_index = np.where(autocorr_timescale_scores >= max_confi)[0][0]

    window_variances = np.var(rolling_window, axis=1)
    variance_valid = np.isclose(abs(window_variances - reference_variance), 0, atol=variance_error)
    variance_scores = np.cumsum(variance_valid[::-1])[::-1] / np.arange(1, len(variance_valid) + 1)[::-1]

    variance_index = np.where(variance_scores >= max_confi)[0][0]

    mean_index = np.where(mean_scores >= max_confi)[0][0]

    stationary_start = max(mean_index, autocorr_timescale_index, variance_index)

    if verbose:
        plt.plot(abs(window_variances - reference_variance), label="var error", color="red")
        plt.hlines(y=(variance_error), xmin=0, xmax=len(window_variances), label="var_err+")
        plt.legend()
        plt.show()

        plt.plot(abs(window_corrs - reference_autocorr_frequency), label="accr error", color="red")
        plt.hlines(y=(autocorr_error), xmin=0, xmax=len(window_corrs), label="accr_err+")
        plt.legend()
        plt.show()

        plt.plot(abs(window_means - reference_mean), label="mean error", color="red")
        plt.hlines(y=(mean_error), xmin=0, xmax=len(window_means), label="mean_err+")
        plt.legend()
        plt.show()

    # Define your time series as an array
    time_series = timeseries[stationary_start:]
    if all(time_series == time_series[0]):
        return stationary_start
    else:

        # Perform the Dickey-Fuller test
        result = adfuller(time_series)

        # Interpret the test results
        critical_value = result[4]['1%']
        if result[0] < critical_value:
            return stationary_start
        else:
            return False


def estimate_error(timeseries, block_size=20, n_samples=4000):
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

    mean_noise = np.zeros(n_samples)
    var_noise = np.zeros(n_samples)
    accr_noise = np.zeros(n_samples)

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

        x_noise = np.random.randn(block_size)
        mean_noise[i] = np.mean(x_noise)
        var_noise[i] = np.var(x_noise)

        freqs, psd = signal.periodogram(x_noise, return_onesided=False)
        accr_noise[i] = freqs[np.argmax(psd)]

    sigma = 12
    mean_jk_error = np.std(mean_jk)
    var_jk_error = np.std(var_jk)
    accr_jk_error = np.std(acorr_jk)

    mean_error = mean_jk_error * sigma
    var_error = var_jk_error * sigma
    accr_error = accr_jk_error * sigma

    return mean_error, var_error, accr_error
