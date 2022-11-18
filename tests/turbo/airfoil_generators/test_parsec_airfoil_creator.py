def test_parsec():
    from ntrfc.turbo.airfoil_generators.parsec_airfoil_creator import parsec_airfoil_gen

    R_LE = 0.01
    x_PRE = 0.450
    y_PRE = -0.006
    d2y_dx2_PRE = -0.4
    th_PRE = 0.05
    x_SUC = 0.450
    y_SUC = 0.055
    d2y_dx2_SUC = -0.350
    th_SUC = -6

    pparray = [R_LE, x_PRE, y_PRE, d2y_dx2_PRE, th_PRE, x_SUC, y_SUC, d2y_dx2_SUC, th_SUC]
    profile_points = parsec_airfoil_gen(pparray)
    x, y = profile_points[::, 0], profile_points[::, 1]
    assert len(x) == len(y)
