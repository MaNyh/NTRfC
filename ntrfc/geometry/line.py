import numpy as np
import pyvista as pv
from scipy.interpolate import CubicSpline,interp1d
from scipy.interpolate import PchipInterpolator


def polyline_from_points(points):
    poly = pv.PolyData()
    poly.points = points
    the_cell = np.arange(0, len(points), dtype=np.int_)
    the_cell = np.insert(the_cell, 0, len(points))
    poly.lines = the_cell
    return poly


def lines_from_points(points):
    """Given an array of points, make a line set"""
    poly = pv.PolyData()
    poly.points = points
    cells = np.full((len(points) - 1, 3), 2, dtype=np.int_)
    cells[:, 1] = np.arange(0, len(points) - 1, dtype=np.int_)
    cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
    poly.lines = cells
    return poly

def refine_spline(x, y, res, half_cosine_spacing=False,kind='linear'):
    """
    Refine spline using cubic interpolation.

    :param x: Array of x coordinates.
    :param y: Array of y coordinates.
    :param res: Number of interpolated points.
    :param half_cosine_spacing: Boolean flag for using half-cosine spacing.
    :return: Interpolated x and y coordinates.
    """

    distance = np.cumsum(np.sqrt(np.ediff1d(x, to_begin=0) ** 2 + np.ediff1d(y, to_begin=0) ** 2))
    distance = distance / distance[-1]

    # Cubic spline interpolation
    fx = interp1d(distance, x,kind=kind)
    fy = interp1d(distance, y,kind=kind)

    if half_cosine_spacing:
        beta = np.linspace(0.0, np.pi, res)
        alpha_ = [(0.5 * (1.0 - np.cos(xa))) for xa in beta]  # Half-cosine based spacing
    else:
        alpha_ = np.linspace(0, 1, res)

    # Interpolated x and y values
    x_new, y_new = fx(alpha_), fy(alpha_)
    return x_new, y_new
