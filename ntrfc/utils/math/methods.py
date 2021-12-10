import numpy as np


def calcAnisoMatrix(R):
    #todo: implement test
    RPG = np.linalg.eigh(R)[1]
    Rii = sum([RPG[0][0], RPG[1][1], RPG[2][2]])
    aniso = R / Rii - np.identity(3) / 3
    return aniso


def calcAnisoEigs(aniso):
    #todo: implement test

    # eigenwerte für anisotropie muss hiermit berechnet werden!
    # wieso nochmal? es tauchen negative eigenwerte auf, bei berechnung mit numpy
    # nicht-symmetrische matrix (anisotrop)
    eigenVal = np.linalg.eig(aniso)
    eigens = list(eigenVal[0])
    eigVec = list(eigenVal[1])
    if list(eigens) == [0, 0, 0]:
        return np.zeros(3), None
    maxEigenIdx = eigens.index(max(eigens))
    minEigenIdx = eigens.index(min(eigens))
    middle = [0, 1, 2]
    middle.remove(maxEigenIdx)
    middle.remove(minEigenIdx)
    middle = middle[0]

    gamma_1 = eigens[maxEigenIdx]
    gamma_2 = eigens[middle]
    gamma_3 = eigens[minEigenIdx]

    eigenVal = np.array([gamma_1, gamma_2, gamma_3])
    return eigenVal, eigVec


def C_barycentric(R):
    #todo: implement test
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


def autocorr(x):
    #todo: implement test
    norm = np.sum(np.array(x) ** 2)
    result = np.correlate(np.array(x), np.array(x), 'full') / norm
    return result[int(len(result) / 2):]


def zero_crossings(data_series):
    #todo: implement test
    zcs = np.where(np.diff(np.sign(data_series)))[0]
    return zcs
