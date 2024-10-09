def test_calc_concavehull():
    """
    in these simple geometries, each point must be found by calcConcaveHull
    """
    from ntrfc.geometry.alphashape import calc_concavehull
    import numpy as np
    import pyvista as pv

    square = pv.Plane()
    boxedges = square.extract_feature_edges()

    boxedges.rotate_z(np.random.randint(0, 360), inplace=True)
    boxpoints = boxedges.points

    np.random.shuffle(boxpoints)

    xs_raw = boxpoints[:, 0]
    ys_raw = boxpoints[:, 1]

    xs, ys = calc_concavehull(xs_raw, ys_raw, 1)

    assert len(xs) == len(xs_raw)
    assert any([yi in ys_raw for yi in ys])

    polygon = pv.Polygon()
    polygon.rotate_z(np.random.randint(0, 360), inplace=True)
    polyedges = polygon.extract_feature_edges()
    polypoints = polyedges.points
    np.random.shuffle(polypoints)
    xs_raw = polypoints[:, 0]
    ys_raw = polypoints[:, 1]

    xs, ys = calc_concavehull(xs_raw, ys_raw, 10)

    assert len(xs) == len(xs_raw)
    assert any([yi in ys_raw for yi in ys])


def test_calc_optimize_alphashape():
    """
    in these simple geometries, each point must be found by calcConcaveHull
    """
    from ntrfc.geometry.alphashape import auto_concaveHull
    import numpy as np
    from ntrfc.turbo.airfoil_generators.naca_airfoil_creator import naca

    for i in range(4):
        digit_string = "6509"

        res = 240
        X, Y = naca(digit_string, res, half_cosine_spacing=True)
        points = np.stack((X[:], Y[:], np.zeros(len(X)))).T * np.random.randn() * 1000

        xs, ys, alpha = auto_concaveHull(points[::, 0], points[::, 1])

        assert abs(len(xs)-len(X)) <= 1
