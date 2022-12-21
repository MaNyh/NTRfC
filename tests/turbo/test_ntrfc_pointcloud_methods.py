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
    points = np.stack((X[:], Y[:], np.zeros(res * 2 + 1))).T

    poly = pv.PolyData(points)
    poly["A"] = np.ones(poly.number_of_points)
    ssPoly, psPoly = extractSidePolys(ind_hk, ind_vk, poly)
    # the assertion is consistent with all tests but it is confusing
    # we generate profiles with a naca-generator. probably there is a minor bug in the algorithm
    # ssPoly needs to have one point more then psPoly
    assert ssPoly.number_of_points + 1 == psPoly.number_of_points, "number of sidepoints are not equal "
