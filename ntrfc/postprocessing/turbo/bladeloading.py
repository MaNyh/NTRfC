from ntrfc.utils.geometry.pointcloud_methods import  extract_profilepoints
from ntrfc.utils.pyvista_utils.plane import areaave_plane
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

def calc_loading_volmesh(volmesh, bladesurface,alpha, verbose=False):
    #todo: test method is missing
    bounds = volmesh.bounds

    zm = (bounds[5]-bounds[4])/2
    midslice_z = bladesurface.slice(origin=(0,0,zm),normal=(0,0,1))

    sortedPoly, psVals, ssVals, ind_vk, ind_hk, midsPoly, beta_leading, beta_trailing, camber_angle= extract_profilepoints(midslice_z,alpha)
    sortedPoints = sortedPoly.points

    x1 = bounds[0] + 1e-5 * bounds[1]
    x2 = bounds[1] + 1e-5 * bounds[0]

    inlet = volmesh.slice(normal="x", origin=(x1, 0, 0)).compute_cell_sizes().point_data_to_cell_data()
    outlet = volmesh.slice(normal="x", origin=(x2, 0, 0)).compute_cell_sizes().point_data_to_cell_data()
    p1 = areaave_plane(inlet,"p")
    p2 = areaave_plane(outlet,"p")

    camber = pv.Line((0, 0, 0), -(sortedPoints[ind_vk] - sortedPoints[ind_hk]))

    shift = sortedPoints[ind_vk]
    shift -= psVals.points[0][-1]

    ssVals.points -= shift
    psVals.points -= shift

    ssVals.rotate_z(-camber_angle)
    psVals.rotate_z(-camber_angle)

    psVals = psVals.cell_data_to_point_data()
    ssVals = ssVals.cell_data_to_point_data()

    ps_xc = np.zeros(psVals.number_of_points)
    ps_cp = np.zeros(psVals.number_of_points)

    for idx, pt in enumerate(psVals.points):
        ps_xc[idx] = pt[0] / camber.length
        ps_cp[idx] = calc_inflow_cp(psVals.point_data["p"][idx], p2, p1)

    ss_xc = np.zeros(ssVals.number_of_points)
    ss_cp = np.zeros(ssVals.number_of_points)

    for idx, pt in enumerate(ssVals.points):
        ss_xc[idx] = pt[0] / camber.length
        ss_cp[idx] = calc_inflow_cp(ssVals.point_data["p"][idx], p2, p1)

    ssVals["xc"] = ss_xc
    ssVals["cp"] = ss_cp
    psVals["xc"] = ps_xc
    psVals["cp"] = ps_cp

    if verbose:
        plt.figure()
        plt.plot(ss_xc, ss_cp)
        plt.plot(ps_xc, ps_cp)
        plt.show()
    return psVals, ssVals

def calc_inflow_cp(px, pt1, p1):
    """
    :param px: pressure at position
    :param pt1: total pressure inlet
    :param p1: pressure inlet
    :return: lift coefficient
    """
    cp = (px - pt1) / (p1 - pt1)
    return cp
