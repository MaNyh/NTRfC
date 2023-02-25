import numpy as np


def calcAnisoMatrix(reynoldsstress_tensor):
    rpg = np.linalg.eigh(reynoldsstress_tensor)[1]
    r_ii = sum([rpg[0][0], rpg[1][1], rpg[2][2]])
    anisotropy_matrix = reynoldsstress_tensor / r_ii - np.identity(3) / 3
    return anisotropy_matrix


def calcAnisoEigs(anisotropy_matrix):
    # eigenwerte für anisotropie muss hiermit berechnet werden!
    # wieso nochmal? es tauchen negative eigenwerte auf, bei berechnung mit numpy
    # nicht-symmetrische matrix (anisotrop)
    eigen_val = np.linalg.eig(anisotropy_matrix)
    eigens = list(eigen_val[0])
    eigen_vec = list(eigen_val[1])
    if list(eigens) == [0, 0, 0]:
        return np.zeros(3), None
    max_eigen_idx = eigens.index(max(eigens))
    min_eigen_idx = eigens.index(min(eigens))
    middle = [0, 1, 2]
    middle.remove(max_eigen_idx)
    middle.remove(min_eigen_idx)
    middle = middle[0]

    gamma_1 = eigens[max_eigen_idx]
    gamma_2 = eigens[middle]
    gamma_3 = eigens[min_eigen_idx]

    eigen_val = np.array([gamma_1, gamma_2, gamma_3])
    return eigen_val, eigen_vec


def C_barycentric(R):
    aniso = calcAnisoMatrix(R)
    anisoEigs = calcAnisoEigs(aniso)[0]
    if list(anisoEigs) == [0, 0, 0]:
        return np.array([0, 0, 1])
    else:
        gamma_1 = anisoEigs[0]
        gamma_2 = anisoEigs[1]
        gamma_3 = anisoEigs[2]

    C1c = gamma_1 - gamma_2
    C2c = 2 * (gamma_2 - gamma_3)
    C3c = 3 * gamma_3 + 1
    CWeights = np.array([C1c, C2c, C3c])

    return CWeights


def autocorr(signal):
    """
    :param: signal - numpy-array.
    """
    norm = np.sum(np.array(signal) ** 2)
    result = np.correlate(np.array(signal), np.array(signal), 'full') / norm
    return result[int(len(result) / 2):]


def zero_crossings(data_series):
    zcs = np.where(np.diff(np.sign(data_series)))[0]
    return zcs


def reldiff(a, b):
    """Calculates the relative difference between two values or arrays of values.

    Parameters:
    a (float or numpy array): The first value or array of values to compare.
    b (float or numpy array): The second value or array of values to compare.

    Returns:
    float or numpy array: The relative difference between a and b. If a and b are both numpy arrays, the output will be a numpy array of the same shape. If one input is a float and the other is a numpy array, the output will be a numpy array of the same shape as the input array.

    Notes:
    The relative difference is defined as the absolute difference between the two values divided by the average of their absolute values.
    If a and b are equal, the relative difference is 0.
    If either a or b is zero, the absolute difference between the two values is returned instead of the relative difference.
    """
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        # Convert to numpy arrays if necessary
        if isinstance(a, float):
            a = np.full(b.shape, a)
        if isinstance(b, float):
            b = np.full(a.shape, b)
        # Calculate relative difference element-wise
        diff = np.abs(a - b) / ((np.abs(a) + np.abs(b)) / 2)
        # Set elements where a or b is zero to the absolute difference
        mask = np.logical_or(a == 0, b == 0)
        diff[mask] = np.abs(a[mask] - b[mask])
        return diff
    else:
        # Handle individual float values
        if a == b:
            return 0
        elif a == 0 or b == 0:
            return abs(a - b)
        else:
            return abs(a - b) / ((abs(a) + abs(b)) / 2)


def return_intersection(hist_1, hist_2):
    """
    Calculate the intersection of two histograms.

    Parameters
    ----------
    hist_1: numpy.ndarray
        The first histogram to be compared.
    hist_2: numpy.ndarray
        The second histogram to be compared.

    Returns
    -------
    float
        The intersection of the two histograms.
    """
    minima = np.minimum(hist_1, hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def is_equidistant(arr, tolerance=1e-8):
    # check if the array has fewer than 2 elements
    if len(arr) < 2:
        return False

    # calculate the difference between the first two elements
    diff = arr[1] - arr[0]

    # create an array of the differences between each element and the next element
    diffs = np.abs(np.diff(arr))

    # use numpy.isclose to check if all of the differences are equal within the tolerance
    return np.all(np.isclose(diffs, diff, atol=tolerance))


def autocorrelate(series):
    """Compute the autocorrelation function of a given time series.

    Args:
        series (array-like): The 1D time series to be autocorrelated.

    Returns:
        array-like: The autocorrelation function of the input series.
    """
    series = series - np.mean(series)
    norm = np.sum(np.array(series) ** 2)
    result = np.correlate(np.array(series), np.array(series), 'full') / norm
    accr = result[int(len(result) / 2):]
    return accr


def minmax_normalize(series_array):
    """Normalize a given 1D array to the range [0, 1] using min-max normalization.

    Args:
        series_array (array-like): The 1D array to be normalized.

    Returns:
        array-like: The normalized 1D array with values in the range [0, 1].

    Note:
        If all values in the input array are the same, the function will return the
        input array unchanged.
    """

    if not np.all(series_array == series_array[0]):
        mins = min(series_array)
        maxs = max(series_array)
        f = (-mins + series_array) / (maxs - mins)
        return f
    else:
        return series_array
