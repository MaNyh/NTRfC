

def test_calc_concavehull():
    """
    in these simple geometries, each point must be found by calcConcaveHull
    """
    from ntrfc.utils.geometry.pointcloud_methods import calc_concavehull
    import numpy as np
    import pyvista as pv

    square = pv.Plane()
    boxedges = square.extract_feature_edges()

    boxedges.rotate_z(np.random.randint(0, 360))
    boxpoints = boxedges.points

    np.random.shuffle(boxpoints)

    xs_raw = boxpoints[:, 0]
    ys_raw = boxpoints[:, 1]

    xs, ys = calc_concavehull(xs_raw, ys_raw, 10)

    assert len(xs) == len(xs_raw)
    assert any([yi in ys_raw for yi in ys])

    polygon = pv.Polygon()
    polygon.rotate_z(np.random.randint(0, 360))
    polyedges = polygon.extract_feature_edges()
    polypoints = polyedges.points
    np.random.shuffle(polypoints)
    xs_raw = polypoints[:, 0]
    ys_raw = polypoints[:, 1]

    xs, ys = calc_concavehull(xs_raw, ys_raw, 10)

    assert len(xs) == len(xs_raw)
    assert any([yi in ys_raw for yi in ys])


def test_parsec():
    from ntrfc.utils.geometry.airfoil_generators.parsec_airfoil_creator import parsec_airfoil_gen

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


def test_naca():
    from ntrfc.utils.geometry.airfoil_generators.naca_airfoil_creator import naca
    import numpy as np

    def rand_naca_code():
        digits = np.random.choice([4, 5])
        if digits == 4:
            d1, d2, d3, d4 = np.random.randint(0, 6), np.random.randint(0, 8), np.random.randint(0,
                                                                                                 6), np.random.randint(
                0, 9)
            digitstring = str(d1) + str(d2) + str(d3) + str(d4)
        if digits == 5:
            d1, d2, d3, d4, d5 = np.random.randint(0, 5), np.random.randint(0, 5), np.random.randint(0,
                                                                                                     6), np.random.randint(
                0, 9), np.random.randint(0, 9)
            digitstring = str(d1) + str(d2) + str(d3) + str(d4) + str(d5)
        return digitstring

    prof_naca = [rand_naca_code() for i in range(6)]
    for i, p in enumerate(prof_naca):
        X, Y = naca(p, 240, finite_te=np.random.choice([True, False]),
                    half_cosine_spacing=np.random.choice([True, False]))


def test_extract_vk_hk(verbose=False):
    """
    tests a NACA  profile in a random angle as a minimal example.
    :return:
    """
    from ntrfc.utils.geometry.pointcloud_methods import extract_vk_hk
    from ntrfc.utils.geometry.airfoil_generators.naca_airfoil_creator import naca
    import numpy as np
    import pyvista as pv

    res = 400

    # d1,d2,d3,d4 = np.random.randint(0,9),np.random.randint(0,9),np.random.randint(0,9),np.random.randint(0,9)
    # digitstring = str(d1)+str(d2)+str(d3)+str(d4)
    # manifold problems with other profiles with veronoi-mid and other unoptimized code. therefor tests only 0009
    # todo: currently we cant test half_cosine_spacing profiles, as the resolution is too good for extract_vk_hk
    X, Y = naca("6506", res, finite_te=False, half_cosine_spacing=False)
    ind_hk_test = 0
    ind_vk_test = res

    points = np.stack((X, Y, np.zeros(len(X)))).T

    profile_points = pv.PolyData(points)

    random_angle = np.random.randint(-40, 40)
    profile_points.rotate_z(random_angle)

    sorted_poly = pv.PolyData(profile_points)
    ind_hk, ind_vk = extract_vk_hk(sorted_poly, verbose=verbose)

    if verbose:
        p = pv.Plotter()
        p.add_mesh(sorted_poly.points[ind_hk], color="yellow", point_size=20)
        p.add_mesh(sorted_poly.points[ind_vk], color="red", point_size=20)
        p.add_mesh(sorted_poly)
        p.show()

    assert (ind_hk == ind_hk_test or ind_hk == ind_vk_test * 2), "wrong hk-index chosen"
    assert ind_vk == ind_vk_test, "wrong vk-index chosen"


def test_midline_from_sides():
    from ntrfc.utils.geometry.pointcloud_methods import midline_from_sides
    from ntrfc.utils.math.vectorcalc import vecAbs
    from ntrfc.utils.geometry.airfoil_generators.naca_airfoil_creator import naca
    from ntrfc.utils.geometry.pointcloud_methods import extractSidePolys
    import numpy as np
    import pyvista as pv

    res = 240
    x, y = naca('0009', res, finite_te=False, half_cosine_spacing=True)
    ind_hk = 0
    ind_vk = res

    points = np.stack((x[:-1], y[:-1], np.zeros(res * 2 - 1))).T
    poly = pv.PolyData(points)
    sspoly, pspoly = extractSidePolys(ind_hk, ind_vk, poly)

    mids = midline_from_sides(ind_hk, ind_vk, poly.points, pspoly, sspoly)

    length = mids.length
    testlength = vecAbs(sspoly.points[0] - sspoly.points[-1])

    assert length == testlength, "midline not accurate"


def test_midLength():
    """
    checks weather
    """
    from ntrfc.utils.geometry.pointcloud_methods import mid_length
    import numpy as np
    import pyvista as pv

    radius = 0.5
    res = 100
    mid = int(res / 2)
    theta = np.linspace(0, 2 * np.pi, 100)

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    fake_vk = 0
    fake_hk = mid
    circle = pv.PolyData(np.stack([x, y, np.zeros(len(x))]).T)
    length = mid_length(fake_vk, fake_hk, circle)
    assert np.isclose(2 * radius, length, rtol=1e-4), "length should be two times the size of the defined test-circle"


def test_extractSidePolys():
    from ntrfc.utils.geometry.pointcloud_methods import extractSidePolys
    from ntrfc.utils.geometry.airfoil_generators.naca_airfoil_creator import naca
    import numpy as np
    import pyvista as pv

    d1, d2, d3, d4 = np.random.randint(0, 9), np.random.randint(0, 9), np.random.randint(0, 9), np.random.randint(0, 9)
    digit_string = str(d1) + str(d2) + str(d3) + str(d4)

    res = 240
    X, Y = naca(digit_string, res, finite_te=False, half_cosine_spacing=True)
    ind_hk = 0
    ind_vk = res
    points = np.stack((X[:-1], Y[:-1], np.zeros(res * 2 - 1))).T

    poly = pv.PolyData(points)
    ssPoly, psPoly = extractSidePolys(ind_hk, ind_vk, poly)
    # todo: this is probably not right. X[1:-1] or X[:-1]? i should not have to use a +1 here
    assert ssPoly.number_of_points == psPoly.number_of_points + 1, "number of sidepoints are not equal "


def test_extract_geo_paras():
    from ntrfc.utils.geometry.pointcloud_methods import extract_profilepoints
    from ntrfc.utils.geometry.airfoil_generators.naca_airfoil_creator import naca
    import numpy as np
    import pyvista as pv

    naca_code = "0009"
    angle = 20  # deg
    alpha = 1
    res = 200
    xs, ys = naca(naca_code, res, finite_te=False, half_cosine_spacing=True)
    sorted_poly = pv.PolyData(np.stack([xs, ys, np.zeros(len(xs))]).T)
    sorted_poly.rotate_z(angle)

    points, ps_poly, ss_poly, ind_vk, ind_hk, mids_poly, beta_leading, beta_trailing, camber_angle = extract_profilepoints(
        sorted_poly, alpha)

    assert np.isclose(beta_leading, (angle + 90)), "wrong leading edge angle"
    assert np.isclose(beta_trailing, (angle + 90)), "wrong leading edge angle"
    assert np.isclose(mids_poly.length, 1)
    assert np.isclose(camber_angle, (angle + 90))
