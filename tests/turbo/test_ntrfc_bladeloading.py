from ntrfc.turbo.bladeloading import calc_cf


def test_calc_inflow_cp():
    from ntrfc.turbo.bladeloading import calc_inflow_cp
    # Test case with positive pressure difference
    px = 101325  # Pa
    pt1 = 110000  # Pa
    p1 = 100000  # Pa
    expected_cp = 0.1325
    assert calc_inflow_cp(px, pt1, p1) == expected_cp

    # Test case with negative pressure difference
    px = 90000  # Pa
    pt1 = 110000  # Pa
    p1 = 100000  # Pa
    expected_cp = -1.0
    assert calc_inflow_cp(px, pt1, p1) == expected_cp

    # Test case with zero pressure difference
    px = 100000  # Pa
    pt1 = 100000  # Pa
    p1 = 110000  # Pa
    expected_cp = 1.0
    assert calc_inflow_cp(px, pt1, p1) == expected_cp


def test_calc_zeta():
    from ntrfc.turbo.bladeloading import calc_zeta
    assert calc_zeta(100, 80, 70) == 0.6666666666666666
    assert calc_zeta(50, 20, 10) == 0.75


def test_calc_cf():
    import numpy as np
    # Test case 1
    tau_w = 0.05
    u_inf = 10
    rho_inf = 1.225
    expected_cf = 0.0008163265306122448
    actual_cf = calc_cf(tau_w, u_inf, rho_inf)
    assert np.isclose(actual_cf, expected_cf)

    # Test case 2
    tau_w = 0.1
    u_inf = 20
    rho_inf = 1.5
    expected_cf = 0.0003333333333333334
    actual_cf = calc_cf(tau_w, u_inf, rho_inf)
    assert np.isclose(actual_cf, expected_cf)
