import os

import numpy as np
import pyvista as pv

ON_CI = 'CI' in os.environ


def test_cascade_3d_domain():
    from ntrfc.turbo.airfoil_generators.naca_airfoil_creator import naca
    from ntrfc.turbo.domaingen_cascade import cascade_3d_domain
    from ntrfc.turbo.domaingen_cascade import cascade_2d_domain

    if ON_CI:
        pv.start_xvfb()

    naca_code = "6509"
    angle = 30  # deg
    res = 420
    xs, ys = naca(naca_code, res, half_cosine_spacing=False)
    sorted_poly = pv.PolyData(np.stack([xs[:-1], ys[:-1], np.zeros(len(xs) - 1)]).T)
    sorted_poly.rotate_z(angle, inplace=True)

    sortedPoly, psPoly, ssPoly, per_y_upper, per_y_lower, inletPoly, outletPoly = cascade_2d_domain(sorted_poly, -1, 2,
                                                                                                    1,
                                                                                                    "m", 0.1, 1,
                                                                                                    path=False)

    cascade_3d_domain(sortedPoly, psPoly, ssPoly, per_y_upper, per_y_lower, inletPoly, outletPoly, zspan=0.2, avdr=1,
                      path=False)


def test_cascade_2d_domain():
    from ntrfc.turbo.airfoil_generators.naca_airfoil_creator import naca
    from ntrfc.turbo.domaingen_cascade import cascade_2d_domain
    if ON_CI:
        pv.start_xvfb()

    naca_code = "6509"
    angle = 30  # deg
    res = 420
    xs, ys = naca(naca_code, res, half_cosine_spacing=False)
    sorted_poly = pv.PolyData(np.stack([xs[:-1], ys[:-1], np.zeros(len(xs) - 1)]).T)
    sorted_poly.rotate_z(angle, inplace=True)
    X, Y = sorted_poly.points[::, 0], sorted_poly.points[::, 1]
    points = np.stack((X[:], Y[:], np.zeros(len(X)))).T
    pointspoly = pv.PolyData(points)
    sortedPoly, psPoly, ssPoly, per_y_upper, per_y_lower, inletPoly, outletPoly = cascade_2d_domain(pointspoly, -1, 2,
                                                                                                    1,
                                                                                                    "m", 0.1, 1, False)
