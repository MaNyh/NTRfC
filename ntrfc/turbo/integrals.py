import numpy as np


def avdr(rho_1, mag_u_1, beta_1, rho_2, mag_u_2, beta_2):
    """
    :param rho_1: float
    :param mag_u_1: float
    :param beta_1: float
    :param rho_2: float
    :param mag_u_2: float
    :param beta_2: float
    :return: float
    """
    if beta_2 == beta_1:
        avdr_res = rho_2 * mag_u_2 / (rho_1 * mag_u_1)
    else:
        avdr_res = rho_2 * mag_u_2 * np.sin(beta_2) / (rho_1 * mag_u_1 * np.sin(beta_1))
    return avdr_res
