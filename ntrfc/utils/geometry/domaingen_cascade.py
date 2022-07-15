import numpy as np
import pyvista as pv
from py2gmsh import (Mesh, Entity, Field)


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

    return sortedPoly, per_y_upper, per_y_lower, inletPoly, outletPoly


def cascade_3d_domain(sortedPoly, per_y_upper, per_y_lower, inletPoly, outletPoly, zspan, avdr=1, verbose=False):

    x_lower =inletPoly.bounds[0]
    x_upper =outletPoly.bounds[0]

    merged_per = per_y_upper.merge(per_y_lower)
    y_lower =merged_per.bounds[2]
    y_upper =merged_per.bounds[3]

    avdr_zdiff_inletoutlet = zspan / 2 * avdr

    domainwidth = abs(y_upper - y_lower)
    # construct a unit-vector-like spline
    per_z_lower = pv.Line((x_lower, y_upper ,zspan / 2 ), (x_upper,y_upper, avdr_zdiff_inletoutlet))
    per_z_upper = pv.Line((x_lower, y_upper, -zspan / 2 ), (x_upper, y_upper, -avdr_zdiff_inletoutlet))

    per_z_lower = per_z_lower.extrude((0, domainwidth, 0))
    per_z_upper = per_z_upper.extrude((0, domainwidth, 0))
    per_z_lower.points-=np.array([0,domainwidth,0])
    per_z_upper.points-=np.array([0,domainwidth,0])

    max_zdiff= 2*avdr_zdiff_inletoutlet
    per_y_upper=polyline_from_points(per_y_upper.points)
    per_y_lower=polyline_from_points(per_y_lower.points)
    sortedPoly=polyline_from_points(np.array([*sortedPoly.points,sortedPoly.points[0]]))
    sortedPoly.translate((0,0,-max_zdiff/2),inplace=True).extrude((0,0,max_zdiff),inplace=True)
    per_y_upper.translate((0,0,-max_zdiff/2),inplace=True).extrude((0,0,max_zdiff),inplace=True)
    per_y_lower.translate((0,0,-max_zdiff/2),inplace=True).extrude((0,0,max_zdiff),inplace=True)
    inletPoly.translate((0,0,-zspan/2),inplace=True).extrude((0,0,zspan),inplace=True)
    outletPoly.translate((0,0,-max_zdiff/2),inplace=True).extrude((0,0,max_zdiff),inplace=True)

    per_y_upper=per_y_upper.clip_surface(per_z_lower)
    per_y_lower=per_y_lower.clip_surface(per_z_lower)
    per_y_upper=per_y_upper.clip_surface(per_z_upper,invert=False)
    per_y_lower=per_y_lower.clip_surface(per_z_upper,invert=False)
    sortedPoly=sortedPoly.clip_surface(per_z_lower)
    sortedPoly=sortedPoly.clip_surface(per_z_upper,invert=False)

    if verbose:
        p = pv.Plotter()
        p.add_mesh(sortedPoly,color="r")
        p.add_mesh(per_y_upper,opacity=0.9)
        p.add_mesh(per_y_lower,opacity=0.9)
        p.add_mesh(inletPoly,opacity=0.9,color="white")
        p.add_mesh(outletPoly,opacity=0.9,color="white")
        # p.add_mesh(per_z_lower,opacity=0.9)
        # p.add_mesh(per_z_upper,opacity=0.9,color="r")
        p.show()

    return sortedPoly,per_y_upper,per_y_lower,inletPoly,outletPoly,per_z_lower,per_z_upper



def gmsh_3d_domain(bladesurface_3d,y_lower_3d,y_upper_3d,inlet_3d,outlet_3d,z_lower_3d,z_upper_3d):
    filename = "bla.geo"
    # create Mesh class instance
    my_mesh = Mesh()
    yper_surface = Entity.SurfaceEntity()
    for cid in range(y_lower_3d.number_of_cells):
        cface = y_lower_3d.extract_cells(cid)
        faces.append(cface)
        cfaceedges = cface.extract_all_edges()
        facecurveset = []
        for eidc in range(cfaceedges.number_of_cells):
            edge=cfaceedges.extract_cells(eidc)
            facecurveset.append(Entity.Curve([edge.points[0], edge.points[0]]))

        ll1 = Entity.CurveLoop(facecurveset, mesh=my_mesh)

        # create surface
        s1 = Entity.PlaneSurface([ll1], mesh=my_mesh)
    return 0
