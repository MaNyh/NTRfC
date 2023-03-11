from ntrfc.geometry.plane import inside_poly


def test_extract_vk_hk(verbose=False):
    """
    tests a NACA  profile in a random angle as a minimal example.
    :return:
    """
    from ntrfc.turbo.profile_tele_extraction import extract_vk_hk
    from ntrfc.turbo.airfoil_generators.naca_airfoil_creator import naca
    import numpy as np
    import pyvista as pv

    # d1,d2,d3,d4 = np.random.randint(0,9),np.random.randint(0,9),np.random.randint(0,9),np.random.randint(0,9)
    # digitstring = str(d1)+str(d2)+str(d3)+str(d4)
    # manifold problems with other profiles with veronoi-mid and other unoptimized code. therefor tests only 0009
    # todo: currently we cant test half_cosine_spacing profiles, as the resolution is too good for extract_vk_hk
    naca_code = "6509"
    angle = 30  # deg
    res = 420
    xs, ys = naca(naca_code, res, half_cosine_spacing=False)
    sorted_poly = pv.PolyData(np.stack([xs[:-1], ys[:-1], np.zeros(len(xs) - 1)]).T)
    sorted_poly.rotate_z(angle, inplace=True)
    X, Y = sorted_poly.points[::, 0], sorted_poly.points[::, 1]
    ind_1 = res
    ind_2 = 0

    points = np.stack((X[:], Y[:], np.zeros(len(X)))).T
    profile_points = pv.PolyData(points)
    sorted_poly = pv.PolyData(profile_points)
    ind_vk, ind_hk = extract_vk_hk(sorted_poly, verbose=verbose)

    if verbose:
        p = pv.Plotter()
        p.add_mesh(sorted_poly.points[ind_hk], color="yellow", point_size=20)
        p.add_mesh(sorted_poly.points[ind_vk], color="red", point_size=20)
        p.add_mesh(sorted_poly)
        p.show()

    assert ind_hk == ind_2, "wrong hk-index chosen"
    assert ind_vk == ind_1, "wrong vk-index chosen"


def test_midline_from_sides(verbose=False):
    from ntrfc.turbo.pointcloud_methods import midline_from_sides
    from ntrfc.math.vectorcalc import vecAbs
    from ntrfc.turbo.airfoil_generators.naca_airfoil_creator import naca
    from ntrfc.turbo.pointcloud_methods import extractSidePolys
    import numpy as np
    import pyvista as pv

    res = 240
    x, y = naca('0009', res, half_cosine_spacing=True)
    ind_hk = 0
    ind_vk = res

    points = np.stack((x[:], y[:], np.zeros(res * 2))).T
    poly = pv.PolyData(points)
    sspoly, pspoly = extractSidePolys(ind_hk, ind_vk, poly)

    mids = midline_from_sides(pspoly, sspoly)

    length = mids.length
    testlength = vecAbs(sspoly.points[0] - sspoly.points[-1])

    if verbose:
        poly["ids"] = np.arange(poly.number_of_points)
        p = pv.Plotter()
        p.add_mesh(poly)
        p.add_mesh(mids)
        p.add_mesh(poly.points[ind_hk], color="k")
        p.add_mesh(poly.points[ind_vk], color="k")
        p.show()

    assert length == testlength, "midline not accurate"


def test_midLength():
    """
    checks weather
    """
    from ntrfc.turbo.pointcloud_methods import mid_length
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


def test_extract_geo_paras(verbose=False):
    from ntrfc.turbo.pointcloud_methods import extract_geo_paras
    from ntrfc.turbo.airfoil_generators.naca_airfoil_creator import naca
    import numpy as np
    import pyvista as pv

    naca_code = "6009"
    angle = 10  # deg
    alpha = 1
    res = 240
    xs, ys = naca(naca_code, res, half_cosine_spacing=False)
    sorted_poly = pv.PolyData(np.stack([xs, ys, np.zeros(len(xs))]).T)
    sorted_poly.rotate_z(angle, inplace=True)

    poly, ps_poly, ss_poly, ind_hk, ind_vk, mids_poly, beta_leading, beta_trailing, camber_angle, alpha = extract_geo_paras(
        sorted_poly, alpha)

    if verbose:
        p = pv.Plotter()
        p.add_mesh(ss_poly, color="g", label="ssPoly")
        p.add_mesh(ps_poly, color="b", label="psPoly")
        p.add_mesh(mids_poly)
        p.add_mesh(poly.points[ind_hk], color="w", label="ind_hk")
        p.add_mesh(poly.points[ind_vk], color="k", label="ind_vk")
        p.add_legend()
        p.show()

    assert np.isclose(beta_leading, angle, rtol=0.03), "wrong leading edge angle"
    assert np.isclose(beta_trailing, angle, rtol=0.03), "wrong leading edge angle"
    assert np.isclose(mids_poly.length, 1, rtol=0.03)
    assert np.isclose(camber_angle, angle, rtol=0.03)
    assert alpha > 0


def test_inside_poly():
    # Test for a simple polygon and point
    polygon = [(0, 0), (0, 1), (1, 1), (1, 0)]
    point = [(0.5, 0.5)]
    assert inside_poly(polygon, point)[0] == True

    # Test for a point outside the polygon
    polygon = [(0, 0), (0, 1), (1, 1), (1, 0)]
    point = [(1.5, 1.5)]
    assert inside_poly(polygon, point)[0] == False

    # Test for a point on the boundary of the polygon
    polygon = [(0, 0), (0, 1), (1, 1), (1, 0)]
    point = [(1, 1)]
    assert inside_poly(polygon, point)[0] == False
