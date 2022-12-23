from ntrfc.turbo.bladeloading import calc_inflow_cp

def test_calc_inflow_cp():
    # Test case with positive pressure difference
    px = 101325  # Pa
    pt1 = 110000  # Pa
    p1 = 100000  # Pa
    expected_cp = 0.8675
    assert calc_inflow_cp(px, pt1, p1) == expected_cp

    # Test case with negative pressure difference
    px = 90000  # Pa
    pt1 = 110000  # Pa
    p1 = 100000  # Pa
    expected_cp = 2
    assert calc_inflow_cp(px, pt1, p1) == expected_cp

    # Test case with zero pressure difference
    px = 100000  # Pa
    pt1 = 100000  # Pa
    p1 = 110000  # Pa
    expected_cp = 0
    assert calc_inflow_cp(px, pt1, p1) == expected_cp
