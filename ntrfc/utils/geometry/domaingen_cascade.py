import numpy as np
import pyvista as pv

from ntrfc.utils.pyvista_utils.line import polyline_from_points
from ntrfc.utils.geometry.pointcloud_methods import extract_geo_paras, calcMidPassageStreamLine


def cascade_2d_domain(profilepoints2d, x_inlet, x_outlet, pitch, unit, blade_shift, alpha, verbose=False):
    """
    profilepoints2d = 2d numpy array
    """
    # todo: implement testmethod!

    # =============================================================================
    # Daten Einlesen
    # =============================================================================
    points = pv.PolyData(np.stack([profilepoints2d[0], profilepoints2d[1], np.zeros(len(profilepoints2d[0]))]).T)
    unitcoeff = 0
    if unit == "m":
        unitcoeff = 1
    elif unit == "mm":
        unitcoeff = 1 / 1000
    points.points *= unitcoeff

    # =============================================================================
    # Bestimmung Profilparameter
    # =============================================================================
    sortedPoly, psPoly, ssPoly, ind_vk, ind_hk, midsPoly, beta_leading, beta_trailing, camber_angle = extract_geo_paras(
        points,
        alpha,
        verbose)
    ##############################################################
    x_mids = midsPoly.points[::, 0]
    y_mids = midsPoly.points[::, 1]
    # x_ss = ssPoly.points[::, 0]
    # y_ss = ssPoly.points[::, 1]
    # x_ps = psPoly.points[::, 0]
    # y_ps = psPoly.points[::, 1]

    x_mpsl, y_mpsl = calcMidPassageStreamLine(x_mids, y_mids, beta_leading, beta_trailing,
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

    return sortedPoly, psPoly, ssPoly,  per_y_upper, per_y_lower, inletPoly, outletPoly


def cascade_3d_domain(sortedPoly, psPoly, ssPoly, per_y_upper, per_y_lower, inletPoly, outletPoly, zspan, avdr=1, verbose=False):

    x_lower =inletPoly.bounds[0]
    x_upper =outletPoly.bounds[0]

    def compute_transform(point,span,avdr,x_lower,x_upper,sign=1):
        l = abs(x_lower-x_upper)
        x = abs(point[0]-x_upper) / l
        return np.array([0,0,sign*span*(1+avdr*x)])+point

    def transform(avdr, poly, x_lower, x_upper, zspan,sign):
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

    if verbose:
        p = pv.Plotter()
        p.add_mesh(psPoly_lowz,color="r")
        p.add_mesh(psPoly_highz,opacity=0.9)
        p.add_mesh(ssPoly_lowz,opacity=0.9)
        p.add_mesh(ssPoly_highz,opacity=0.9,color="white")
        p.add_mesh(per_y_upper_lowz,opacity=0.9,color="white")
        p.add_mesh(per_y_upper_highz,opacity=0.9,color="white")
        p.add_mesh(per_y_lower_lowz,opacity=0.9,color="white")
        p.add_mesh(per_y_lower_highz,opacity=0.9,color="white")
        p.add_mesh(inletPoly_lowz,opacity=0.9,color="white")
        p.add_mesh(inletPoly_highz,opacity=0.9,color="white")
        p.add_mesh(outletPoly_lowz,opacity=0.9,color="white")
        p.add_mesh(outletPoly_highz,opacity=0.9,color="white")
        p.show()

    return sortedPoly_lowz,sortedPoly_high,per_y_upper_lowz,per_y_upper_highz,\
           per_y_lower_lowz,per_y_lower_highz,\
           inletPoly_lowz,inletPoly_highz,\
           outletPoly_lowz,outletPoly_highz







