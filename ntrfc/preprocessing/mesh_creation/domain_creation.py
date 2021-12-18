from ntrfc.utils.geometry.pointcloud_methods import calcMidPassageStreamLine
from ntrfc.utils.pyvista_utils.line import lines_from_points
import numpy as np
import pyvista as pv
import os

def create_2d_domain(geosettings, basedir, midsPoly, ssPoly, psPoly, geometry_paras):
    beta_meta_01 = geometry_paras["beta_meta_01"]
    beta_meta_02 = geometry_paras["beta_meta_02"]
    x_inlet = geosettings["x_inlet"]
    x_outlet = geosettings["x_outlet"]
    pitch = geosettings["pitch"]
    blade_shift = geosettings["blade_shift"]
    x_mids = midsPoly.points[::, 0]
    y_mids = midsPoly.points[::, 1]
    x_ss = ssPoly.points[::, 0]
    y_ss = ssPoly.points[::, 1]
    x_ps = psPoly.points[::, 0]
    y_ps = psPoly.points[::, 1]

    stagger_angle = np.rad2deg(np.arctan((y_mids[-1] - y_mids[-0]) / (x_mids[-1] - x_mids[-0])))

    x_mpsl, y_mpsl = calcMidPassageStreamLine(x_mids, y_mids, beta_meta_01, beta_meta_02,
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

    domain_dir = os.path.join(basedir,"02_boundaries")
    inletPoly.save(os.path.join(domain_dir,"inlet_2d.vtk"),False)
    outletPoly.save(os.path.join(domain_dir,"outlet_2d.vtk"),False)
    per_y_upper.save(os.path.join(domain_dir,"y_upper_2d.vtk"),False)
    per_y_lower.save(os.path.join(domain_dir,"y_lower_2d.vtk"),False)

def create_2d_blocklines(geo_settings,basedir,sortedPoints,geometry_paras,midsPoly,inlet,outlet,y_upper,y_lower):
    bladeline = lines_from_points(sortedPoints)
    bladehelpersurface = bladeline.extrude((0, 0, 0.01))
    ogridhelpersurface = bladehelpersurface.compute_normals()
    #todo: replace .002 with a propper generic scaling (using the length of the profile and the distance to a periodic as reference)
    ogridhelpersurface.points += .002 * ogridhelpersurface.point_normals
    ogridline=ogridhelpersurface.slice(normal="z")
    ogridline.points[:, 2] = 0

    p=pv.Plotter()
    p.add_mesh(inlet)
    p.add_mesh(inlet)
    p.add_mesh(y_upper)
    p.add_mesh(y_lower)
    p.add_mesh(bladeline)

    p.add_mesh(ogridline)

    p.show()
    return 0
