import numpy as np

from ntrfc.math.methods import find_nearest
from ntrfc.timeseries.integral_scales import get_self_correlating_frequencies


def min_max_normalize(signal):
    """Normalize a signal using min-max normalization.

    This function scales the input signal so that its minimum value maps to 0 and its maximum value maps to 1.

    Parameters:
    signal (array-like): The signal to be normalized.

    Returns:
    numpy.ndarray: The normalized signal.

    Example:
    array([0. , 0.5, 1. ])
    """
    max_val = np.max(signal)

    if max_val == 0:
        return signal
    normalized_signal = (signal - min(signal)) / (max(signal) - min(signal))

    return normalized_signal


def classify_stationarity(timesteps, values):
    """Classify a stationary time series signal as constant, noise, or deterministic.

    Parameters:
    timesteps (array-like): A list of timesteps corresponding to the time series signal.
    values (array-like): A list of values corresponding to the time series signal at the given timesteps.

    Returns:
    tuple: A tuple of three boolean values indicating whether the time series signal is classified as constant, noise, or deterministic.

    Example:
    classify_stationarity([1, 2, 3, 4], [0, 0, 0, 0]) -> (True, False, False)
    classify_stationarity([1, 2, 3, 4], [0, 1, 0, 1]) -> (False, False, True)
    classify_stationarity([1, 2, 3, 4], [1, 2, 3, 4]) -> (False, True, False)
    """

    normed_values = min_max_normalize(values)

    deterministic_signal = recreate_signal(timesteps, normed_values)
    noise = normed_values - deterministic_signal

    transformed_normed_signal = np.fft.fft(normed_values)

    energy_normed = np.sum(np.abs(np.fft.fft(transformed_normed_signal)) ** 2)
    std_noise = np.std(noise)
    std_deterministic = np.std(deterministic_signal)

    if energy_normed == 0:
        const_bool = True
        deterministic_bool = False
        noise_bool = False
    else:
        const_bool = False
        deterministic_bool = True if std_noise < std_deterministic else False
        noise_bool = True if std_deterministic < std_noise else False

    return const_bool, deterministic_bool, noise_bool


def recreate_signal(timesteps, signal):
    # Find self-correlating frequencies
    mean = np.mean(signal)
    signal_zeroed = signal - mean

    self_correlating_frequencies = get_self_correlating_frequencies(timesteps, signal_zeroed)
    # Decompose the signal into its frequency components using the FFT

    frequencies = np.fft.fftfreq(len(timesteps), d=timesteps[1] - timesteps[0])
    nearfreqs = np.unique([find_nearest(frequencies, i) for i in self_correlating_frequencies])

    transformed = np.fft.fft(signal_zeroed)
    select = [True if i not in nearfreqs else False for i in range(len(transformed))]
    transformed[select] = complex(0)

    recreated = np.real(np.fft.ifft(transformed))

    # Calculate the energy of the signals
    energy_1 = np.sum(np.abs(transformed) ** 2)
    energy_2 = np.sum(np.abs(np.fft.fft(recreated)) ** 2)

    scaling = energy_1 / energy_2
    recreated = recreated * scaling + mean

    return recreated
