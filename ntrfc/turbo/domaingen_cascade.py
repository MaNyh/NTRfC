import tempfile

import numpy as np
import pyvista as pv

from ntrfc.turbo.cascade_case.utils.domain_utils import Blade2D
from ntrfc.turbo.pointcloud_methods import calcMidPassageStreamLine


def cascade_2d_domain(profilepoints2d, x_inlet, x_outlet, pitch, unit, blade_shift, alpha, path=False):
    """
    profilepoints2d = 2d numpy array
    """

    if path is None:
        tmpdir = tempfile.mkdtemp()
        path = tmpdir + "/plot.png"
    # =============================================================================
    # Daten Einlesen
    # =============================================================================
    unitcoeff = 0
    if unit == "m":
        unitcoeff = 1
    elif unit == "mm":
        unitcoeff = 1 / 1000
    profilepoints2d.points *= unitcoeff

    blade = Blade2D(profilepoints2d.points)
    blade.compute_all_frompoints()

    x_mpsl, y_mpsl = calcMidPassageStreamLine(blade.skeletonline_pv.points[::, 0], blade.skeletonline_pv.points[::, 1],
                                              blade.beta_le, blade.beta_te,
                                              x_inlet, x_outlet, pitch)
    y_upper = np.array(y_mpsl) + blade_shift
    per_y_upper = pv.lines_from_points(np.stack((np.array(x_mpsl),
                                                 np.array(y_upper),
                                                 np.zeros(len(x_mpsl)))).T)
    y_lower = y_upper - pitch
    per_y_lower = pv.lines_from_points(np.stack((np.array(x_mpsl),
                                                 np.array(y_lower),
                                                 np.zeros(len(x_mpsl)))).T)

    inlet_pts = np.array([per_y_lower.points[per_y_lower.points[::, 0].argmin()],
                          per_y_upper.points[per_y_upper.points[::, 0].argmin()]])

    inletPoly = pv.Line(*inlet_pts)
    outlet_pts = np.array([per_y_lower.points[per_y_lower.points[::, 0].argmax()],
                           per_y_upper.points[per_y_upper.points[::, 0].argmax()]])

    outletPoly = pv.Line(*outlet_pts)

    p = pv.Plotter(off_screen=True)
    p.add_mesh(outletPoly, color="r")
    p.add_mesh(inletPoly, color="r")
    p.add_mesh(per_y_lower, color="r")
    p.add_mesh(per_y_upper, color="r")
    p.add_mesh(profilepoints2d, color="k")
    p.view_xy()
    p.screenshot(path)

    return blade.sortedpoints_pv, blade.ps_pv, blade.ss_pv, per_y_upper, per_y_lower, inletPoly, outletPoly


def cascade_3d_domain(sortedPoly, psPoly, ssPoly, per_y_upper, per_y_lower, inletPoly, outletPoly, zspan, avdr=1,
                      path=None):
    if path is None:
        tmpdir = tempfile.mkdtemp()
        path = tmpdir + "/plot.png"

    x_lower = inletPoly.bounds[0]
    x_upper = outletPoly.bounds[0]

    def compute_transform(point, span, avdr, x_lower, x_upper, sign=1):
        lval = abs(x_lower - x_upper)
        x = abs(point[0] - x_upper) / lval
        return np.array([0, 0, sign * span * (1 + avdr * x)]) + point

    def transform(avdr, poly, x_lower, x_upper, zspan, sign):
        poly_copy = poly.copy()
        for idx, pt in enumerate(poly_copy.points):
            poly_copy.points[idx] = compute_transform(pt, zspan, avdr, x_lower, x_upper, sign)
        return poly_copy

    sortedPoly_lowz = transform(avdr, sortedPoly, x_lower, x_upper, zspan, -1)
    sortedPoly_high = transform(avdr, sortedPoly, x_lower, x_upper, zspan, 1)

    psPoly_lowz = transform(avdr, psPoly, x_lower, x_upper, zspan, -1)
    psPoly_highz = transform(avdr, psPoly, x_lower, x_upper, zspan, 1)

    ssPoly_lowz = transform(avdr, ssPoly, x_lower, x_upper, zspan, -1)
    ssPoly_highz = transform(avdr, ssPoly, x_lower, x_upper, zspan, 1)

    per_y_upper_lowz = transform(avdr, per_y_upper, x_lower, x_upper, zspan, -1)
    per_y_upper_highz = transform(avdr, per_y_upper, x_lower, x_upper, zspan, 1)

    per_y_lower_lowz = transform(avdr, per_y_lower, x_lower, x_upper, zspan, -1)
    per_y_lower_highz = transform(avdr, per_y_lower, x_lower, x_upper, zspan, 1)

    inletPoly_lowz = transform(avdr, inletPoly, x_lower, x_upper, zspan, -1)
    inletPoly_highz = transform(avdr, inletPoly, x_lower, x_upper, zspan, 1)

    outletPoly_lowz = transform(avdr, outletPoly, x_lower, x_upper, zspan, -1)
    outletPoly_highz = transform(avdr, outletPoly, x_lower, x_upper, zspan, 1)

    p = pv.Plotter(off_screen=True)
    p.add_mesh(psPoly_lowz, color="r")
    p.add_mesh(psPoly_highz, opacity=0.9)
    p.add_mesh(ssPoly_lowz, opacity=0.9)
    p.add_mesh(ssPoly_highz, opacity=0.9, color="white")
    p.add_mesh(per_y_upper_lowz, opacity=0.9, color="white")
    p.add_mesh(per_y_upper_highz, opacity=0.9, color="white")
    p.add_mesh(per_y_lower_lowz, opacity=0.9, color="white")
    p.add_mesh(per_y_lower_highz, opacity=0.9, color="white")
    p.add_mesh(inletPoly_lowz, opacity=0.9, color="white")
    p.add_mesh(inletPoly_highz, opacity=0.9, color="white")
    p.add_mesh(outletPoly_lowz, opacity=0.9, color="white")
    p.add_mesh(outletPoly_highz, opacity=0.9, color="white")
    p.view_xy()
    p.screenshot(path)

    return sortedPoly_lowz, sortedPoly_high, per_y_upper_lowz, per_y_upper_highz, \
           per_y_lower_lowz, per_y_lower_highz, \
           inletPoly_lowz, inletPoly_highz, \
           outletPoly_lowz, outletPoly_highz
