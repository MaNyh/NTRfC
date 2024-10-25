import numpy as np
import pandas as pd
from scipy import signal
from scipy.spatial.distance import jensenshannon

from ntrfc.math.methods import minmax_normalize


def optimal_bin_width(sample1, sample2):
    """
    Compute the optimal bin width using cross-validation estimator for mean integrated squared error.

    Parameters:
    -----------
    data : numpy array
        One-dimensional array of data points.

    Returns:
    --------
    h : float
        Optimal bin width.
    """

    data = np.concatenate([sample1, sample2])
    if np.all(data == data[0]):
        return [data[0] - 0.5, data[0] + 0.5]
    n = len(data)
    h = 2 * np.std(data) * n ** (-1 / 3)  # initial bin width using Scott's rule

    # perform cross-validation to estimate mean integrated squared error
    J_min = np.inf
    for i in range(12):
        bins = np.arange(np.min(data), np.max(data) + h, h)
        counts, _ = np.histogram(data, bins=bins)
        N_k = counts[np.nonzero(counts)]
        J = 2 / ((n - 1) * h) - (n + 1) / (n ** 2 * (n - 1) * h) * np.sum(N_k ** 2)
        if J < J_min:
            J_min = J
            h_opt = h
        h *= 0.8  # decrease bin width for better accuracy
    bins = np.arange(min(np.min(sample1), np.min(sample2)), max(np.max(sample1), np.max(sample2)), h_opt)
    return bins


def smd_probability_compare(sample1, sample2, verbose=False):
    """Compare the probability distribution of two signals using the Freedman-Diaconis rule
    to determine the number of bins.

    Args:
        sample1 (numpy.ndarray): First signal.
        sample2 (numpy.ndarray): Second signal.

    Returns:
        float: Mean squared error between the probability distribution densities
            of the two signals. A value of 0 indicates that the probability distributions
            are not alike, while a value of 1 indicates that they are equal.
    """
    # Compute the number of bins using the Freedman-Diaconis rule.
    bins = optimal_bin_width(sample1, sample2)

    # Compute the histogram and probability density of the first signal.
    hist1, _ = np.histogram(sample1, bins=bins, density=True)
    pdf1 = hist1 / np.sum(hist1)

    # Compute the histogram and probability density of the second signal.
    hist2, _ = np.histogram(sample2, bins=bins, density=True)
    pdf2 = hist2 / np.sum(hist2)

    # Compute the mean squared error between the probability densities.
    mse = np.sum((pdf1 - pdf2) ** 2)
    # Convert the mse to a similarity score between 0 and 1.
    similarity = 1 - mse
    # if verbose:
    #     # Plot the histogram
    #     plt.hist(sample1, bins=bins)
    #     plt.hist(sample2, bins=bins)
    #     plt.show()
    return similarity

def calculate_histogram(signal, bins=30):
    # Calculate histogram as an approximation of the distribution
    hist, bin_edges = np.histogram(signal, bins=bins, density=True)
    hist = hist / hist.sum()  # Normalize to make it a probability distribution
    return hist

def calculate_stationarity_score(sample_1, sample_2):

    bins = optimal_bin_width(sample_1, sample_2)

    def js_divergence_with_reference(sample1,sample2,bins):
        hist1 = calculate_histogram(sample1, bins)
        hist2 = calculate_histogram(sample2, bins)
        return jensenshannon(hist1, hist2)

    # Apply rol
    # Calculate JS divergence for each rolling window compared to the reference
    js_divergences = js_divergence_with_reference(sample_1,sample_2,bins)

    # Calculate average distance
    avg_distance = js_divergences
    score = avg_distance  # Lower distance means more staionary

    return score


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

    # Normalize the time series
    normalized_series = minmax_normalize(time_series)

    # Get the length of the time series and define a range of allowed window sizes
    series_length = len(normalized_series)
    min_window_size = int(series_length * min_interval)
    max_window_size = int(series_length * max_interval)
    allowed_window_sizes = range(min_window_size, max_window_size)

    def rdiff(a,b):
        if a != 0:
            return np.abs((a-b)/a)
        elif b != 0:
            return np.abs((a-b)/b)
        else:
            return 0

    scores = []
    mean_scores = []
    var_scores = []
    for window_size in allowed_window_sizes:
        window_1 = normalized_series[-window_size:]
        window_2 = normalized_series[-window_size * 2:-window_size]
        mean_scores.append(rdiff(window_1.mean(), window_2.mean()))
        var_scores.append(rdiff(window_1.var(), window_2.var()))
        # Compute the correlation coefficient
        scores.append(calculate_stationarity_score(window_1, window_2))

    mean_scores = np.array(mean_scores)
    var_scores = np.array(var_scores)

    cumulated_scores = minmax_normalize(mean_scores) + minmax_normalize(var_scores)
    optimal_window_size_index = np.argmin(cumulated_scores)
    opt_window_size = allowed_window_sizes[optimal_window_size_index]*2

    #
    opt_window = time_series[-opt_window_size:]

    assert len(opt_window) == opt_window_size

    if scores[optimal_window_size_index] > 0.1:
        return False, False, False

    # Compute the period of the time series
    freqs, psd = signal.welch(opt_window.T, fs=1, nperseg=opt_window_size // 2)
    maxpsdid = np.argmax(psd)

    if maxpsdid != 0:
        tperiod = freqs[np.argmax(psd)] ** -1
        nperiods = (opt_window_size + (-opt_window_size % tperiod)) // tperiod
    else:
        tperiod = np.inf
        nperiods = 0

    # If verbose mode is enabled, display a plot of the correlation coefficient and KS test results for each window size
    # if verbose:
    #     plt.plot(cumulated_scores)
    #     plt.axvline(optimal_window_size_index)
    #     plt.legend()
    #     plt.show()

    return opt_window, opt_window_size, nperiods


def estimate_stationarity(timeseries, verbose=False):
    sigma_threshold = 3
    normalized_series = minmax_normalize(timeseries)  # minmax_normalize(timeseries)
    datalength = len(normalized_series)
    opt_window, opt_window_size, nperiods = optimal_window_size(normalized_series)
    if not opt_window_size:
        return False

    reference_window = opt_window

    reference_mean = np.mean(reference_window)
    reference_variance = np.var(reference_window)
    opt_rolling_window = pd.DataFrame(reference_window).rolling(window=opt_window_size//2)

    rolling_means = opt_rolling_window.mean().values
    rolling_vars = opt_rolling_window.var().values

    mean_uncertainty = np.nanstd(rolling_means)
    var_uncertainty = np.nanstd(rolling_vars)


    mean_tolerance = (0.005)
    var_tolerance = 0.15*mean_tolerance



    checkseries = normalized_series[:-opt_window_size]

    checkseries_reversed = pd.DataFrame(checkseries[::-1])
    rolling_win_reversed = checkseries_reversed.rolling(window=opt_window_size//2)

    rolling_means_reversed = rolling_win_reversed.mean().values
    rolling_vars_reversed = rolling_win_reversed.var().values

    rolling_means_errors_reversed = np.abs(rolling_means_reversed - reference_mean)
    rolling_vars_errors_reversed = np.abs(rolling_vars_reversed - reference_variance)

    mean_limits = sigma_threshold * (mean_uncertainty) + mean_tolerance
    var_limits = sigma_threshold * (var_uncertainty) + var_tolerance

    rolling_means_errors_inliers_reversed = rolling_means_errors_reversed[opt_window_size:] <= mean_limits
    rolling_vars_errors_inliers_reversed = rolling_vars_errors_reversed[opt_window_size:] <= var_limits

    def last_coherent_interval(arr, threshold=0.99):
        reversed_arr = arr[::-1]
        false_indices = np.where(reversed_arr == False)[0]
        last_false_index = false_indices[-1] if len(false_indices) > 0 else 0
        coherent_arr = [False] * last_false_index + [True] * (len(reversed_arr) - last_false_index)
        success_rate = np.array([np.sum(coherent_arr[i:]) / len(coherent_arr[i:]) for i in range(len(coherent_arr))])
        answer_index = np.where(success_rate >= threshold)[0][0]
        return len(arr) - answer_index

    mean_index = datalength - last_coherent_interval(rolling_means_errors_inliers_reversed) - 3 * opt_window_size//2

    variance_index = datalength - last_coherent_interval(rolling_vars_errors_inliers_reversed) - 3 * opt_window_size//2

    stationary_start_index = max(mean_index, variance_index)-opt_window_size//2

    # if verbose:
    #     fig, axs = plt.subplots(3, 1, figsize=(24, 20))
    #     axs[0].plot(normalized_series, label="normalized series", color="blue")
    #     axs[0].vlines(x=stationary_start_index, ymin=0, ymax=np.nanmax(normalized_series), label="stationary_start",
    #                   color="k")
    #
    #     axs[1].plot(np.nan_to_num(rolling_means_errors_reversed[::-1], 0), label="mean error", color="red")
    #     axs[1].hlines(y=(mean_limits), xmin=0, xmax=len(normalized_series), label="mean_limits", color="k")
    #     axs[1].vlines(x=stationary_start_index, ymin=0, ymax=max(mean_limits, np.nanmax(rolling_means_errors_reversed)),
    #                   label="stationary_start", color="k")
    #     axs[1].vlines(x=mean_index, ymin=0, ymax=max(mean_limits, np.nanmax(rolling_means_errors_reversed)),
    #                   label="mean_index", color="green")
    #     axs[1].legend()
    #
    #     axs[2].plot(np.nan_to_num(rolling_vars_errors_reversed[::-1], 0), label="variance error", color="red")
    #     axs[2].hlines(y=(var_limits), xmin=0, xmax=len(normalized_series), label="var_limits", color="k")
    #     axs[2].vlines(x=stationary_start_index, ymin=0, ymax=max(var_limits, np.nanmax(rolling_vars_errors_reversed)),
    #                   label="stationary_start", color="k")
    #     axs[2].vlines(x=variance_index, ymin=0, ymax=max(var_limits, np.nanmax(rolling_vars_errors_reversed)),
    #                   label="variance_index", color="k")
    #     axs[2].legend()
    #
    #     plt.show()

    return stationary_start_index


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
