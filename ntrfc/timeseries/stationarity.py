import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from sklearn.decomposition import PCA
from statsmodels.tsa.arima.model import ARIMA

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


def var_compare(sample1, sample2):
    var1 = np.var(sample1)
    var2 = np.var(sample2)

    if var1 == var2:
        return 1.1  # variance is exactly the same, return value greater than 1
    else:
        diff = np.abs(var1 - var2)
        max_var = max(var1, var2)
        return max(0, 1 - diff / max_var)


def pd_compare(sample1, sample2, verbose=False):
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
    if verbose:
        # Plot the histogram
        plt.hist(sample1, bins=bins)
        plt.hist(sample2, bins=bins)
        plt.show()
    return similarity


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

    # Get the length of the time series and define a range of allowed window sizes
    series_length = len(normalized_series)
    allowed_window_sizes = np.array(range(series_length // 10, series_length // 4))

    # Define a threshold for the sanity test
    sanity_threshold = 2.82

    # Iterate through the allowed window sizes and perform the Kolmogorov-Smirnov test and correlation coefficient calculation
    corr_coeff_scores = np.zeros(len(allowed_window_sizes))
    pd_similiarity_scores = np.zeros(len(allowed_window_sizes))
    var_similiarity_scores = np.zeros(len(allowed_window_sizes))
    for i, window_size in enumerate(allowed_window_sizes):
        check_window = normalized_series[-window_size * 2:]
        first_subsequence = check_window[:-window_size]
        second_subsequence = check_window[-window_size:]
        pd_similiarity = pd_compare(second_subsequence, first_subsequence)  #
        corr_coeff = (1 + np.corrcoef(first_subsequence, second_subsequence)[0][1]) / 2
        var_similiarity = var_compare(first_subsequence, second_subsequence)
        if np.array_equal(first_subsequence, first_subsequence):
            corr_coeff = 1
        corr_coeff_scores[i] = corr_coeff
        pd_similiarity_scores[i] = pd_similiarity
        var_similiarity_scores[i] = var_similiarity
    cumulated_scores = corr_coeff_scores + pd_similiarity_scores + var_similiarity_scores
    if not any(cumulated_scores >= sanity_threshold):
        return False, False, False

    optimal_window_size_index = np.argmax(cumulated_scores)
    opt_window_size = allowed_window_sizes[optimal_window_size_index]

    opt_window = time_series[series_length - opt_window_size * 2:]
    freqs, psd = signal.periodogram(opt_window, return_onesided=False)
    maxpsdid = np.argmax(psd)
    if maxpsdid != 0:
        tperiod = freqs[np.argmax(psd)] ** -1
        nperiods = int(opt_window_size / tperiod)
    else:
        tperiod = np.inf
        nperiods = 0

    # If verbose mode is enabled, display a plot of the correlation coefficient and KS test results for each window size
    if verbose:
        plt.plot(corr_coeff_scores)
        plt.plot(pd_similiarity_scores)
        plt.axvline(optimal_window_size_index)
        plt.legend()
        plt.show()

    return opt_window, opt_window_size, nperiods


def estimate_seasonal_error(series, opt_window_size):
    snr_winsize = int(np.ceil((opt_window_size) / 16))
    if snr_winsize >= 4:
        reconstructed_mean, fluct, snr = snr_pod(series, snr_winsize)
    else:
        reconstructed_mean, fluct, snr = snr_pod(series, len(series) // 2)

    # Calculate the rolling window of residuals
    rolling_residuals = np.lib.stride_tricks.sliding_window_view(reconstructed_mean, opt_window_size)

    # Calculate the dominant frequency of residuals for each window
    residual_freqs = []
    for window in rolling_residuals:
        freqs, psd = signal.periodogram(window, return_onesided=False)
        dominant_freq = freqs[np.argmax(psd)]
        residual_freqs.append(dominant_freq)

    # Calculate the standard deviation of mean residuals, residual variance, and dominant frequencies
    residual_var = np.std(np.var(rolling_residuals, axis=1))
    residual_mean = np.std(np.mean(rolling_residuals, axis=1))
    residual_freq = np.std(residual_freqs)

    return residual_mean, residual_var, residual_freq


def estimate_stationarity(timeseries, verbose=False):
    sigma_threshold = 4
    normalized_series = minmax_normalize(timeseries)
    opt_window, opt_window_size, nperiods = optimal_window_size(normalized_series)
    if not opt_window_size:
        return False

    reference_window = opt_window
    reference_freqs, reference_psd = signal.periodogram(reference_window, return_onesided=False)

    reference_mean = np.mean(reference_window)
    reference_autocorr_freq = reference_freqs[np.argmax(reference_psd)]
    reference_variance = np.var(reference_window)

    residual_mean, residual_variance, residual_autocorr = estimate_residual_error(reference_window, opt_window_size)
    seasonal_mean, seasonal_variance, seasonal_autocorr = estimate_seasonal_error(reference_window,
                                                                                  opt_window_size // nperiods)

    mean_jk_error, var_jk_error, accr_jk_error = estimate_error_jacknife(reference_window,
                                                                         block_size=opt_window_size // nperiods)
    modelerror_mean, modelerror_freq, modelerror_var = estimate_model_error(reference_window, opt_window_size, nperiods)

    autocorr_limit = (seasonal_autocorr + residual_autocorr + accr_jk_error + modelerror_freq) * sigma_threshold
    mean_limit = (seasonal_mean + residual_mean + mean_jk_error + modelerror_mean) * sigma_threshold
    variance_limit = (seasonal_variance + residual_variance + var_jk_error + modelerror_var) * sigma_threshold

    rolling_window = np.lib.stride_tricks.sliding_window_view(normalized_series, opt_window_size)

    def cum_score(valid):
        scores = np.cumsum(valid[::-1])[::-1] / np.arange(1, len(valid) + 1)[::-1]
        index = np.where(scores >= 1)[0][0]
        return scores, index

    window_means = np.mean(rolling_window, axis=1)
    mean_valid = np.isclose(abs(window_means - reference_mean), 0, atol=mean_limit)
    if not any(mean_valid):
        return False
    mean_scores, mean_index = cum_score(mean_valid)

    window_corrs = []
    for window in rolling_window:
        freqs, psd = signal.periodogram(window, return_onesided=False)
        window_corrs.append(freqs[np.argmax(psd)])
    autocorr_timescale_valid = np.isclose(abs(window_corrs - reference_autocorr_freq), 0, atol=autocorr_limit)
    if not any(autocorr_timescale_valid):
        return False
    autocorr_timescale_scores, autocorr_timescale_index = cum_score(autocorr_timescale_valid)

    window_variances = np.var(rolling_window, axis=1)
    variance_valid = np.isclose(abs(window_variances - reference_variance), 0, atol=variance_limit)
    if not any(variance_valid):
        return False
    variance_scores, variance_index = cum_score(variance_valid)

    stationary_start_index = max(mean_index, autocorr_timescale_index, variance_index)
    if verbose:
        plt.plot(abs(window_variances - reference_variance), label="var error", color="red")
        plt.hlines(y=(variance_limit), xmin=0, xmax=len(window_variances), label="var_err+")
        plt.legend()
        plt.show()

        plt.plot(abs(window_corrs - reference_autocorr_freq), label="accr error", color="red")
        plt.hlines(y=(autocorr_limit), xmin=0, xmax=len(window_corrs), label="accr_err+")
        plt.legend()
        plt.show()

        plt.plot(abs(window_means - reference_mean), label="mean error", color="red")
        plt.hlines(y=(mean_limit), xmin=0, xmax=len(window_means), label="mean_err+")
        plt.legend()
        plt.show()

    return stationary_start_index


def estimate_residual_error(reference, opt_window_size):
    """
    Estimates the residual errors for a given time series using an ARIMA model and a rolling window approach.

    In the context of time series analysis, the residual error (or simply "residuals") represents the difference
    between the actual values of the time series and the predicted values from a model. In other words, it is the
    part of the time series that the model is unable to explain.

    Args:
        reference (array-like): A time series to estimate residual errors for.
        opt_window_size (int): The size of the rolling window to use for calculating metrics.

    Returns:
        Tuple: A tuple containing the following metrics of the residual errors:
            - residual_mean (float): The standard deviation of the mean residuals over the rolling windows.
            - residual_var (float): The standard deviation of the residual variance over the rolling windows.
            - residual_freq (float): The standard deviation of the dominant frequencies of the residuals over the rolling windows.
    """
    # Fit an ARIMA model to the reference data
    arima_model = ARIMA(reference, order=(1, 1, 1))
    arima_results = arima_model.fit()

    # Calculate the residuals
    residuals = reference - arima_results.fittedvalues

    # Calculate the rolling window of residuals
    rolling_residuals = np.lib.stride_tricks.sliding_window_view(residuals, opt_window_size)

    # Calculate the dominant frequency of residuals for each window
    residual_freqs = []
    for window in rolling_residuals:
        freqs, psd = signal.periodogram(window, return_onesided=False)
        dominant_freq = freqs[np.argmax(psd)]
        residual_freqs.append(dominant_freq)

    # Calculate the standard deviation of mean residuals, residual variance, and dominant frequencies
    residual_var = np.std(np.var(rolling_residuals, axis=1))
    residual_mean = np.std(np.mean(rolling_residuals, axis=1))
    residual_freq = np.std(residual_freqs)

    return residual_mean, residual_var, residual_freq


def estimate_model_error(reference_window, opt_window_size, nper):
    ref_length = len(reference_window)
    ref_mean = np.mean(reference_window)
    ref_var = np.var(reference_window)

    # Calculate the dominant frequency of residuals for each window
    freqs, psd = signal.periodogram(reference_window, return_onesided=False)
    ref_freq = freqs[np.argmax(psd)]

    x = np.linspace(0, 2 * np.pi * nper, ref_length)

    y = np.sin(x) + 1

    rolling_residuals = np.lib.stride_tricks.sliding_window_view(y, opt_window_size)
    # Calculate the dominant frequency of residuals for each window
    residual_ts = []
    for window in rolling_residuals:
        freqs, psd = signal.periodogram(window, return_onesided=False)
        dominant_freq = freqs[np.argmax(psd)]
        residual_ts.append(dominant_freq)

    y_mean = 1
    y_var = 0.5
    y_t = nper / ref_length

    # Calculate the standard deviation of mean residuals, residual variance, and dominant frequencies
    residual_var = np.var(rolling_residuals, axis=1)
    residual_mean = np.mean(rolling_residuals, axis=1)
    residual_freq = residual_ts

    residual_length = len(residual_mean)
    mean_error = np.mean(np.abs(residual_mean - np.array([y_mean] * residual_length))) / y_mean * ref_mean
    var_error = np.mean(np.abs(residual_var - np.array([y_var] * residual_length))) / y_var * ref_var
    freq_error = np.mean(np.abs(residual_freq - np.array([y_t] * residual_length))) / y_t * ref_freq
    return mean_error, freq_error, var_error


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


def snr_pod(x, window_size, verbose=False):
    ncomp = 16
    if ncomp > window_size:
        # fails otherwise
        ncomp = window_size
    X = np.zeros((x.shape[0] - window_size, window_size))
    for i in range(X.shape[0]):
        X[i] = x[i:i + window_size]
    # Apply PCA to the windowed data
    pca = PCA(n_components=ncomp)
    pca.fit(X)
    # Reconstruct the data using the principal components
    reconstructed = pca.inverse_transform(pca.transform(X))
    reconstructed_mean = reconstructed.mean(axis=1)
    fluct = x[window_size // 2:-window_size // 2] - reconstructed_mean
    snr = np.trapz(reconstructed_mean ** 2) / np.trapz(fluct ** 2)
    if verbose:
        plt.plot(fluct)
        plt.plot(reconstructed_mean)
        plt.show()
    return reconstructed_mean, fluct, snr
