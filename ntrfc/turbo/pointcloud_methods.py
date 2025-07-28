import numpy as np
import pyvista as pv

from ntrfc.geometry.line import refine_spline, polyline_from_points
from typing import Literal


def midline_from_sides(ps_poly, ss_poly, mode: Literal["2D", "Q3D"], res=100):
    x_ps, y_ps = ps_poly.points[::, 0], ps_poly.points[::, 1]
    x_ss, y_ss = ss_poly.points[::, 0], ss_poly.points[::, 1]

    if mode == "2D":
        z_ss, z_ps = np.zeros_like(x_ss), np.zeros_like(x_ps)
    elif mode == "3D":
        z_ss, z_ps = ss_poly.points[::, 2], ps_poly.points[::, 2]
    else:
        raise ValueError("Mode must be either '2D' or 'Q3D'.")

    midsres = res
    if x_ps[0] > x_ps[-1]:
        ax, ay, az = refine_spline(x_ps[::-1], y_ps[::-1], z_ps[::-1], midsres, kind="linear")
    else:
        ax, ay, az = refine_spline(x_ps, y_ps, z_ps, midsres, kind="linear")
    if x_ss[0] > x_ss[-1]:
        bx, by, bz = refine_spline(x_ss[::-1], y_ss[::-1], z_ss[::-1], midsres, kind="linear")
    else:
        bx, by, bz = refine_spline(x_ss, y_ss, z_ss, midsres, kind="linear")

    xmids, ymids, zmids = ((ax + bx) / 2, (ay + by) / 2, (az + bz) / 2)

    midsPoly = polyline_from_points(np.stack((xmids, ymids, zmids)).T)
    
    return midsPoly


def extractSidePolys(ind_1, ind_2, sortedPoly):
    # xs, ys = list(sortedPoly.points[::, 0]), list(sortedPoly.points[::, 1])
    indices = np.arange(0, sortedPoly.number_of_points)

    if ind_2 > ind_1:
        side_one_idx = indices[ind_1:ind_2 + 1]
        side_two_idx = np.concatenate((indices[:ind_1 + 1][::-1], indices[ind_2:][::-1]))
    elif ind_1 > ind_2:
        side_one_idx = indices[ind_2:ind_1 + 1]
        side_two_idx = np.concatenate((indices[:ind_2 + 1][::-1], indices[ind_1:][::-1]))

    side_one = pv.PolyData()
    for i in side_one_idx:
        side_one = side_one.merge(sortedPoly.extract_points(i))

    side_two = pv.PolyData()
    for i in side_two_idx:
        side_two = side_two.merge(sortedPoly.extract_points(i))

    if side_one.center[1]<side_two.center[1]:
        return side_one, side_two
    else:
        return side_two, side_one


def is_counterclockwise(xs, ys):
    n = len(xs)
    signed_area = 0

    for i in range(n):
        x1, y1 = xs[i], ys[i]
        x2, y2 = xs[(i + 1) % n], ys[(i + 1) % n]  # Wrap around to the first point for the last segment
        signed_area += (x1 * y2 - x2 * y1)

    return signed_area > 0


def calcMidPassageStreamLine(x_mcl, y_mcl, beta1, beta2, x_inlet, x_outlet, t):
    """
    Calculate the midpoint stream line curve through a passage.

    Parameters:
    -----------
    x_mcl : array_like
        The x-coordinates of the mid-chord line.
    y_mcl : array_like
        The y-coordinates of the mid-chord line.
    beta1 : float
        The angle in degrees representing the inflow angle at the inlet.
    beta2 : float
        The angle in degrees representing the outflow angle at the outlet.
    x_inlet : float
        The x-coordinate of the inlet position.
    x_outlet : float
        The x-coordinate of the outlet position.
    t : float
        The pitch of the midpoint stream line.
    verbose : bool, optional
        If True, a plot of the midpoint stream line is displayed.

    Returns:
    --------
    x_mpsl_ref : array_like
        The refined x-coordinates of the midpoint stream line.
    y_mpsl_ref : array_like
        The refined y-coordinates of the midpoint stream line.
    """

    delta_x_vk = x_mcl[0] - x_inlet
    delta_y_vk = np.tan(np.deg2rad(beta1)) * delta_x_vk

    p_inlet_x = x_mcl[0] - delta_x_vk
    p_inlet_y = y_mcl[0] - delta_y_vk

    delta_x_hk = x_outlet - x_mcl[-1]
    delta_y_hk = delta_x_hk * np.tan(np.deg2rad(beta2))

    p_outlet_x = x_mcl[-1] + delta_x_hk
    p_outlet_y = y_mcl[-1] + delta_y_hk

    x_mpsl = [p_inlet_x] + list(x_mcl) + [p_outlet_x]
    y_mpsl = [p_inlet_y] + list(y_mcl) + [p_outlet_y]

    for i in range(len(x_mpsl)):
        y_mpsl[i] = y_mpsl[i] + 0.5 * t

    x_mpsl_ref, y_mpsl_ref = refine_spline(x_mpsl, y_mpsl, 1000)

    return x_mpsl_ref, y_mpsl_ref
