import numpy as np

from ntrfc.turbo.pointcloud_methods import midline_from_sides


def test_Blade2d_extractSidePolys():
    from ntrfc.turbo.cascade_case.utils.domain_utils import Blade2D
    from ntrfc.turbo.airfoil_generators.naca_airfoil_creator import naca
    import numpy as np
    import pyvista as pv

    digit_string = "0009"

    res = 480
    X, Y = naca(digit_string, res, half_cosine_spacing=False, finite_te=False)

    points = np.stack((X[:], Y[:], np.zeros(len(X)))).T

    poly = pv.PolyData(points)
    poly["A"] = np.ones(poly.number_of_points)
    blade = Blade2D(poly)

    blade.compute_all_frompoints()
    # the assertion is consistent with all tests but it is confusing
    # we generate profiles with a naca-generator. probably there is a minor bug in the algorithm
    # ssPoly needs to have one point more then psPoly
    assert abs(
        blade.ss_pv.number_of_points - blade.ps_pv.number_of_points) / res < 0.02, "number of sidepoints are not equal "
    assert np.all(blade.ss_pv["A"])
    assert np.all(blade.ps_pv["A"])


def test_extractSidePolys():
    import pyvista as pv
    from ntrfc.turbo.pointcloud_methods import extractSidePolys

    res = 100

    circlepoly = pv.PolyData(pv.Circle(radius=1, resolution=res).points)

    ind_1 = 0
    ind_2 = circlepoly.number_of_points // 2
    side_1, side_2 = extractSidePolys(ind_1, ind_2, circlepoly)

    assert side_1.number_of_points == res // 2 + 1
    assert side_2.number_of_points == res // 2 + 1


def test_calcMidPassageStreamLine():
    from ntrfc.turbo.pointcloud_methods import calcMidPassageStreamLine

    # Define input values
    x_mcl = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_mcl = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    beta1 = 10.0
    beta2 = 5.0
    x_inlet = 0.0
    x_outlet = 6.0
    t = 0.1

    # Calculate actual output
    x_mpsl_ref, y_mpsl_ref = calcMidPassageStreamLine(x_mcl, y_mcl, beta1, beta2, x_inlet, x_outlet, t)

    # Test output
    assert len(x_mpsl_ref) == 1000
    assert len(y_mpsl_ref) == 1000


def test_extractSidePolys():
    import pyvista as pv
    from ntrfc.turbo.pointcloud_methods import extractSidePolys

    res = 100

    circlepoly = pv.PolyData(np.roll(pv.Circle(radius=1, resolution=res).points, res // 2, axis=0))
    ind_1 = 0
    ind_2 = circlepoly.number_of_points // 2
    side_1, side_2 = extractSidePolys(ind_1,ind_2, circlepoly)

    assert side_1.number_of_points == res // 2 + 1
    assert side_2.number_of_points == res // 2 + 1

    assert np.all(side_1.points[0] == side_2.points[0])
    assert np.all(side_1.points[-1] == side_2.points[-1])


def test_midline_from_sides():
    import pyvista as pv
    from ntrfc.turbo.pointcloud_methods import extractSidePolys

    res = 100

    circlepoly = pv.PolyData(np.roll(pv.Circle(radius=1, resolution=res).points, res // 2, axis=0))

    ind_1 = 0
    ind_2 = circlepoly.number_of_points // 2
    side_1, side_2 = extractSidePolys(ind_1, ind_2, circlepoly)

    mids = midline_from_sides(side_1, side_2)

    assert np.allclose(mids.points[:, 1], 0, atol=1e-14)
