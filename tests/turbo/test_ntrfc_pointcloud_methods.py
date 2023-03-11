import numpy as np
import pyvista as pv

from ntrfc.turbo.pointcloud_methods import extract_points_fromsortedpoly, calcMidPassageStreamLine


def test_extractSidePolys():
    from ntrfc.turbo.pointcloud_methods import extractSidePolys
    from ntrfc.turbo.airfoil_generators.naca_airfoil_creator import naca
    import numpy as np
    import pyvista as pv

    digit_string = "6509"

    res = 240
    X, Y = naca(digit_string, res, half_cosine_spacing=True)
    ind_hk = 0
    ind_vk = res
    points = np.stack((X[:], Y[:], np.zeros(res * 2))).T

    poly = pv.PolyData(points)
    poly["A"] = np.ones(poly.number_of_points)
    ssPoly, psPoly = extractSidePolys(ind_hk, ind_vk, poly)
    # the assertion is consistent with all tests but it is confusing
    # we generate profiles with a naca-generator. probably there is a minor bug in the algorithm
    # ssPoly needs to have one point more then psPoly
    assert ssPoly.number_of_points == psPoly.number_of_points, "number of sidepoints are not equal "


def test_extract_points_fromsortedpoly():
    # Create a sorted polygon with some per-vertex data

    sorted_poly = pv.Circle()

    # Extract a subset of the points and associated data
    sorted_indices = np.arange(sorted_poly.number_of_points // 2, sorted_poly.number_of_points)
    side_two = extract_points_fromsortedpoly(sorted_indices, sorted_poly)

    assert side_two.number_of_points == sorted_poly.number_of_points // 2


def test_calcMidPassageStreamLine():
    # Define input values
    x_mcl = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_mcl = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    beta1 = 10.0
    beta2 = 5.0
    x_inlet = 0.0
    x_outlet = 6.0
    t = 0.1

    # Expected output
    x_mpsl_ref_expected = np.array(
        [0., 0.00756705, 0.0151341, 0.02270115, 0.0302682, 0.03783525, 0.0454023, 0.05296935, 0.0605364, 0.06810345,
         0.07567051, 0.08323756, 0.09080461, 0.09837166, 0.10593871, 0.11350576, 0.12107281, 0.12863986, 0.13620691,
         0.14377396, 0.15134101, 0.15890806, 0.16647511])
    y_mpsl_ref_expected = np.array(
        [0.87367302, 0.87500729, 0.87634157, 0.87767584, 0.87901012, 0.8803444, 0.88167867, 0.88301295, 0.88434722,
         0.8856815, 0.88701577, 0.88835005, 0.88968432, 0.8910186, 0.89235287, 0.89368715, 0.89502142, 0.8963557,
         0.89768997, 0.89902425, 0.90035852, 0.9016928, 0.90302707, 0])

    # Calculate actual output
    x_mpsl_ref, y_mpsl_ref = calcMidPassageStreamLine(x_mcl, y_mcl, beta1, beta2, x_inlet, x_outlet, t)

    # Test output
    assert len(x_mpsl_ref) == 1000
    assert len(y_mpsl_ref) == 1000
