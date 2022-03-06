import numpy as np


def calcAnisoMatrix(reynoldsstress_tensor):
    # todo: implement test
    rpg = np.linalg.eigh(reynoldsstress_tensor)[1]
    r_ii = sum([rpg[0][0], rpg[1][1], rpg[2][2]])
    anisotropy_matrix = reynoldsstress_tensor / r_ii - np.identity(3) / 3
    return anisotropy_matrix


def calcAnisoEigs(anisotropy_matrix):
    # todo: implement test

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
    # todo: implement test
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
    # todo: implement test
    norm = np.sum(np.array(signal) ** 2)
    result = np.correlate(np.array(signal), np.array(signal), 'full') / norm
    return result[int(len(result) / 2):]


def zero_crossings(data_series):
    # todo: implement test
    zcs = np.where(np.diff(np.sign(data_series)))[0]
    return zcs
