import numpy as np
import pyvista as pv


def test_polyline_from_points():
    from ntrfc.geometry.line import polyline_from_points

    points = pv.Line(resolution=100).points
    line = polyline_from_points(points)
    assert line.length == 1.0, "theres something fishy about the polyline_from_points implementation"


def test_line_from_points():
    from ntrfc.geometry.line import lines_from_points

    points = pv.Line(resolution=100).points
    line = lines_from_points(points)
    assert line.length == 1.0, "theres something fishy about the polyline_from_points implementation"


def test_refine_spline():
    """
    tests if you can refine a spline by checking the number of points and the length of the spline
    """
    from ntrfc.geometry.line import refine_spline

    coarseres = 2
    line = pv.Line(resolution=coarseres)
    fineres = 100
    fline_xx, fline_yy = refine_spline(line.points[::, 0], line.points[::, 1], fineres)
    fline = pv.lines_from_points(np.stack([fline_xx, fline_yy, np.zeros(len(fline_xx))]).T)
    assert line.length == fline.length
    assert fline.number_of_points == fineres

    coarseres = 2
    line = pv.Line(resolution=coarseres)
    fineres = 100
    fline_xx, fline_yy = refine_spline(line.points[::, 0], line.points[::, 1], fineres, half_cosine_spacing=True)
    fline = pv.lines_from_points(np.stack([fline_xx, fline_yy, np.zeros(len(fline_xx))]).T)
    assert line.length == fline.length
    assert fline.number_of_points == fineres
