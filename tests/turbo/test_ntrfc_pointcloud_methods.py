import numpy as np


def test_extractSidePolys():
    from ntrfc.turbo.cascade_case.utils.domain_utils import Blade2D
    from ntrfc.turbo.airfoil_generators.naca_airfoil_creator import naca
    import numpy as np
    import pyvista as pv

    digit_string = "0009"

    res = 120
    X, Y = naca(digit_string, res, half_cosine_spacing=False, finite_te=False)

    points = np.stack((X[:], Y[:], np.zeros(len(X)))).T

    poly = pv.PolyData(points)
    poly["A"] = np.ones(poly.number_of_points)
    blade = Blade2D(poly)

    blade.compute_all_frompoints()
    # the assertion is consistent with all tests but it is confusing
    # we generate profiles with a naca-generator. probably there is a minor bug in the algorithm
    # ssPoly needs to have one point more then psPoly
    assert blade.ss_pv.number_of_points == blade.ps_pv.number_of_points, "number of sidepoints are not equal "
    assert np.all(blade.ss_pv["A"])
    assert np.all(blade.ps_pv["A"])


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
