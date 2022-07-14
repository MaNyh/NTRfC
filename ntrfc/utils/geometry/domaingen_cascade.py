import numpy as np
import pyvista as pv
from ntrfc.utils.geometry.pointcloud_methods import extract_geo_paras, calcMidPassageStreamLine

def cascade_domain(profilepoints2d,x_inlet,x_outlet,pitch,unit,blade_shift,alpha,span_z, verbose=False):
    """
    profilepoints2d = 2d numpy array
    """
    # =============================================================================
    # Daten Einlesen
    # =============================================================================
    points = pv.PolyData(np.stack([profilepoints2d[0],profilepoints2d[1],np.zeros(len(profilepoints2d[0]))]).T)
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

    return sortedPoly, per_y_upper, per_y_lower, inletPoly, outletPoly
