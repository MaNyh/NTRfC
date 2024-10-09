def test_naca():
    from ntrfc.turbo.airfoil_generators.naca_airfoil_creator import naca
    import numpy as np
    import pyvista as pv
    res = 480

    xs, ys = naca("6510", res, half_cosine_spacing=False, finite_te=False)
    points = np.stack([xs, ys, np.zeros(len(xs))]).T
    poly = pv.PolyData(points)
    assert poly.number_of_points == res * 2 + 1

    xs, ys = naca("6510", res, half_cosine_spacing=True, finite_te=False)
    points = np.stack([xs, ys, np.zeros(len(xs))]).T
    poly = pv.PolyData(points)
    assert poly.number_of_points == res * 2 + 1

    xs, ys = naca("6510", res, half_cosine_spacing=False, finite_te=True)
    points = np.stack([xs, ys, np.zeros(len(xs))]).T
    poly = pv.PolyData(points)
    assert poly.number_of_points == res * 2 - 1 + 100

    xs, ys = naca("6510", res, half_cosine_spacing=True, finite_te=True)
    points = np.stack([xs, ys, np.zeros(len(xs))]).T
    poly = pv.PolyData(points)
    assert poly.number_of_points == res * 2 - 1 + 100

    xs, ys = naca("23112", res, half_cosine_spacing=False, finite_te=False)
    points = np.stack([xs, ys, np.zeros(len(xs))]).T
    poly = pv.PolyData(points)
    assert poly.number_of_points == res * 2 + 1

    xs, ys = naca("23112", res, half_cosine_spacing=True, finite_te=False)
    points = np.stack([xs, ys, np.zeros(len(xs))]).T
    poly = pv.PolyData(points)
    assert poly.number_of_points == res * 2 + 1

    xs, ys = naca("23112", res, half_cosine_spacing=False, finite_te=True)
    points = np.stack([xs, ys, np.zeros(len(xs))]).T
    poly = pv.PolyData(points)
    assert poly.number_of_points == res * 2 - 1 + 100

    xs, ys = naca("23112", res, half_cosine_spacing=True, finite_te=True)
    points = np.stack([xs, ys, np.zeros(len(xs))]).T
    poly = pv.PolyData(points)
    assert poly.number_of_points == res * 2 - 1 + 100
