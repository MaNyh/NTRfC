import numpy as np

from ntrfc.turbo.integrals import avdr


def test_avdr():
    rho_1 = 1
    mag_u_1 = 1
    beta_1 = np.pi
    rho_2 = 1
    mag_u_2 = 1
    beta_2 = np.pi

    ans = avdr(rho_1, mag_u_1, beta_1, rho_2, mag_u_2, beta_2)
    assert ans == 1, "error"

    rho_1 = 1
    mag_u_1 = 10
    beta_1 = np.pi
    rho_2 = 1
    mag_u_2 = 20
    beta_2 = np.pi

    ans = avdr(rho_1, mag_u_1, beta_1, rho_2, mag_u_2, beta_2)
    assert ans == 2, "error"

    rho_1 = 1
    mag_u_1 = 10
    beta_1 = np.pi / 4
    rho_2 = 1
    mag_u_2 = 4.082485
    beta_2 = np.pi / 3

    ans = avdr(rho_1, mag_u_1, beta_1, rho_2, mag_u_2, beta_2)
    assert ans == 0.5000002566283092, "error"
