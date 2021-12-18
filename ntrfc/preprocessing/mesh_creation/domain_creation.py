from ntrfc.utils.geometry.pointcloud_methods import calcMidPassageStreamLine
from ntrfc.utils.pyvista_utils.line import lines_from_points
from ntrfc.utils.math.vectorcalc import closest_node_index, vecAbs, vecDir
from ntrfc.utils.filehandling.tecplot import writeTecplot1DFile
import numpy as np
import pyvista as pv
import os
import matplotlib.pyplot as plt


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

    domain_dir = os.path.join(basedir, "02_domainboundaries")
    inletPoly.save(os.path.join(domain_dir, "inlet_2d.vtk"), False)
    outletPoly.save(os.path.join(domain_dir, "outlet_2d.vtk"), False)
    per_y_upper.save(os.path.join(domain_dir, "y_upper_2d.vtk"), False)
    per_y_lower.save(os.path.join(domain_dir, "y_lower_2d.vtk"), False)

    plt.figure()
    plt.plot(inletPoly.points[::, 0], inletPoly.points[::, 1], color="#6c3376")
    plt.plot(outletPoly.points[::, 0], outletPoly.points[::, 1], color="#FF2211")
    plt.plot(per_y_upper.points[::, 0], per_y_upper.points[::, 1], color="#FF22CC")
    plt.plot(per_y_lower.points[::, 0], per_y_lower.points[::, 1], color="#FF22CC")
    plt.plot(x_ss, y_ss, color="#FFAA44")
    plt.plot(x_ps, y_ps, color="#FFAA44")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(os.path.join(domain_dir, 'domain.pdf'))


def create_2d_blocklines(geo_settings, basedir, sortedPoints, psPoly, ssPoly, geometry_paras, midsPoly, inlet, outlet,
                         y_upper, y_lower):
    bladeline = lines_from_points(sortedPoints)
    bladehelpersurface = bladeline.extrude((0, 0, 0.01))

    ogridhelpersurface = bladehelpersurface.compute_normals()

    # compute size of ogrid. there are probably better solutions
    ogrid_zero = bladeline.points[0]
    upper_idx = closest_node_index(ogrid_zero, y_upper.points)
    lower_idx = closest_node_index(ogrid_zero, y_lower.points)
    distance_upper = vecAbs(y_upper.points[upper_idx] - ogrid_zero)
    distance_lower = vecAbs(y_lower.points[lower_idx] - ogrid_zero)
    ogrid_size = 0.3 * min([distance_upper, distance_lower])
    chord_start = 0.05
    chord_idx_helper = int(chord_start * midsPoly.number_of_points)
    cstart_1 = midsPoly.points[chord_idx_helper]
    cstart_2 = midsPoly.points[-chord_idx_helper]
    ogridhelpersurface.points += ogrid_size * ogridhelpersurface.point_normals
    ogridline = ogridhelpersurface.slice(normal="z")
    ogridline.points[:, 2] = 0
    domain_dir = os.path.join(basedir,"03_meshgeometry")


    blademids = midsPoly.copy()
    # blademids.points += pitch/2

    ps_x0_blade = closest_node_index(cstart_1, psPoly.points)
    ps_x1_blade = closest_node_index(cstart_2, psPoly.points)

    ss_x0_blade = closest_node_index(cstart_1, ssPoly.points)
    ss_x1_blade = closest_node_index(cstart_2, ssPoly.points)

    ps_x0_ogrid_idx = closest_node_index(psPoly.points[ps_x0_blade], ogridline.points)
    ps_x1_ogrid_idx = closest_node_index(psPoly.points[ps_x1_blade], ogridline.points)

    ss_x0_ogrid_idx = closest_node_index(ssPoly.points[ss_x0_blade], ogridline.points)
    ss_x1_ogrid_idx = closest_node_index(ssPoly.points[ss_x1_blade], ogridline.points)

    ps_x0_ogrid_line = pv.Line(ogridline.points[ps_x0_ogrid_idx], psPoly.points[ps_x0_blade])
    ps_x1_ogrid_line = pv.Line(ogridline.points[ps_x1_ogrid_idx], psPoly.points[ps_x1_blade])
    ss_x0_ogrid_line = pv.Line(ogridline.points[ss_x0_ogrid_idx], ssPoly.points[ss_x0_blade])
    ss_x1_ogrid_line = pv.Line(ogridline.points[ss_x1_ogrid_idx], ssPoly.points[ss_x1_blade])

    yperlow_x0_ps_idx = np.argmin((cstart_1[0]-y_lower.points[::,0])**2)
    yperlow_x1_ps_idx = np.argmin((cstart_2[0]-y_lower.points[::,0])**2)

    ylower_ogrid_x0 = pv.Line(y_lower.points[yperlow_x0_ps_idx], ogridline.points[ss_x0_ogrid_idx])
    yupper_ogrid_x0 = pv.Line(y_upper.points[yperlow_x0_ps_idx], ogridline.points[ps_x0_ogrid_idx])

    ylower_ogrid_x1 = pv.Line(y_lower.points[yperlow_x1_ps_idx], ogridline.points[ss_x1_ogrid_idx])
    yupper_ogrid_x1 = pv.Line(y_upper.points[yperlow_x1_ps_idx], ogridline.points[ps_x1_ogrid_idx])


    msp_xx,msp_yy = calcMidPassageStreamLine(midsPoly.points[::,0],midsPoly.points[::,1],
                                   geometry_paras["beta_meta_01"],
                                   geometry_paras["beta_meta_02"],
                                   inlet.bounds[0],outlet.bounds[0],0)

    mspPoly = lines_from_points(np.stack([msp_xx,msp_yy,np.zeros(len(msp_yy))]).T)

    le_ogrid = pv.Line(midsPoly.points[0],midsPoly.points[0]-vecDir(midsPoly.points[1]-midsPoly.points[0])*ogrid_size)
    te_ogrid = pv.Line(midsPoly.points[-1],midsPoly.points[-1]-vecDir(midsPoly.points[-2]-midsPoly.points[-1])*ogrid_size)
    ogrid_inlet_dist = vecAbs(mspPoly.points[0]-le_ogrid.points[-1])
    ogrid_outlet_dist = vecAbs(mspPoly.points[-1]-te_ogrid.points[-1])
    ogrid_inlet = pv.Line(le_ogrid.points[-1],le_ogrid.points[-1]-vecDir(midsPoly.points[1]-midsPoly.points[0])*ogrid_inlet_dist)
    ogrid_oulet = pv.Line(te_ogrid.points[-1],te_ogrid.points[-1]-vecDir(midsPoly.points[-2]-midsPoly.points[-1])*ogrid_outlet_dist)
    """
    ogridline.save(os.path.join(domain_dir, "ogridline_2d.vtk"))


    ogrid_oulet.save(os.path.join(domain_dir, "ogrid_oulet.vtk"))
    ogrid_inlet.save(os.path.join(domain_dir, "ogrid_inlet.vtk"))
    te_ogrid.save(os.path.join(domain_dir, "te_ogrid.vtk"))
    le_ogrid.save(os.path.join(domain_dir, "le_ogrid.vtk"))
    yupper_ogrid_x1.save(os.path.join(domain_dir, "yupper_ogrid_x1.vtk"))
    ylower_ogrid_x1.save(os.path.join(domain_dir, "ylower_ogrid_x1.vtk"))
    yupper_ogrid_x0.save(os.path.join(domain_dir, "yupper_ogrid_x0.vtk"))
    ylower_ogrid_x0.save(os.path.join(domain_dir, "ylower_ogrid_x0.vtk"))
    ss_x1_ogrid_line.save(os.path.join(domain_dir, "ss_x1_ogrid_line.vtk"))
    ss_x0_ogrid_line.save(os.path.join(domain_dir, "ss_x0_ogrid_line.vtk"))
    ps_x1_ogrid_line.save(os.path.join(domain_dir, "ps_x1_ogrid_line.vtk"))
    ps_x0_ogrid_line.save(os.path.join(domain_dir, "ps_x0_ogrid_line.vtk"))
    """
    plt.figure()
    plt.plot(inlet.points[::, 0], inlet.points[::, 1], color="#6c3376")
    plt.plot(outlet.points[::, 0], outlet.points[::, 1], color="#FF2211")
    plt.plot(y_upper.points[::, 0], y_upper.points[::, 1], color="#FF22CC")
    plt.plot(y_lower.points[::, 0], y_lower.points[::, 1], color="#FF22CC")
    plt.plot(bladeline.points[::, 0], bladeline.points[::, 1], color="#6c3376")
    plt.plot(le_ogrid.points[::, 0], le_ogrid.points[::, 1], color="#FF2211")
    plt.plot(te_ogrid.points[::, 0], te_ogrid.points[::, 1], color="#FF22CC")
    plt.plot(ogrid_inlet.points[::, 0], ogrid_inlet.points[::, 1], color="#FF22CC")
    plt.plot(ogrid_oulet.points[::, 0], ogrid_oulet.points[::, 1], color="#6c3376")
    plt.plot(ogridline.points[::, 0], ogridline.points[::, 1], color="#FF2211")
    plt.plot(ps_x0_ogrid_line.points[::, 0], ps_x0_ogrid_line.points[::, 1], color="#FF22CC")
    plt.plot(ps_x1_ogrid_line.points[::, 0], ps_x1_ogrid_line.points[::, 1], color="#FF22CC")
    plt.plot(ss_x0_ogrid_line.points[::, 0], ss_x0_ogrid_line.points[::, 1], color="#FF22CC")
    plt.plot(ss_x1_ogrid_line.points[::, 0], ss_x1_ogrid_line.points[::, 1], color="#FF22CC")
    plt.plot(ylower_ogrid_x0.points[::, 0], ylower_ogrid_x0.points[::, 1], color="#FF22CC")
    plt.plot(yupper_ogrid_x0.points[::, 0], yupper_ogrid_x0.points[::, 1], color="#FF22CC")
    plt.plot(ylower_ogrid_x1.points[::, 0], ylower_ogrid_x1.points[::, 1], color="#FF22CC")
    plt.plot(yupper_ogrid_x1.points[::, 0], yupper_ogrid_x1.points[::, 1], color="#FF22CC")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(os.path.join(domain_dir, 'blocklines.pdf'))


    writeTecplot1DFile(os.path.join(domain_dir,"domain.geom"),['x', 'z'],
                       ["inlet", "outlet", "y_upper", "y_lower", "psPoly", "ssPoly"],
                       [[inlet.points[::,0],inlet.points[::,1]],[outlet.points[::,0],outlet.points[::,1]],
                        [y_upper.points[::,0],y_upper.points[::,1]],[y_lower.points[::,0],y_lower.points[::,1]],
                        [psPoly.points[::,0],psPoly.points[::,1]],[ssPoly.points[::,0],ssPoly.points[::,1]]],
                       "domainboundaries")

    writeTecplot1DFile(os.path.join(domain_dir,"blocks.geom"),['x', 'z'],
                       ["le_ogrid","te_ogrid","ogrid_inlet","ogrid_oulet","ogridline",
                       "ps_x0_ogrid_line","ps_x1_ogrid_line","ss_x0_ogrid_line","ss_x1_ogrid_line",
                       "ylower_ogrid_x0","yupper_ogrid_x0","ylower_ogrid_x1","yupper_ogrid_x1"],
                       [[le_ogrid.points[::,0],le_ogrid.points[::,1]],
                        [te_ogrid.points[::,0],te_ogrid.points[::,1]],
                        [ogrid_inlet.points[::,0],ogrid_inlet.points[::,1]],
                        [ogrid_oulet.points[::,0],ogrid_oulet.points[::,1]],
                        [ogridline.points[::,0],ogridline.points[::,1]],
                        [ps_x0_ogrid_line.points[::,0],ps_x0_ogrid_line.points[::,1]],
                        [ps_x1_ogrid_line.points[::,0],ps_x1_ogrid_line.points[::,1]],
                        [ss_x0_ogrid_line.points[::,0],ss_x0_ogrid_line.points[::,1]],
                        [ss_x1_ogrid_line.points[::,0],ss_x1_ogrid_line.points[::,1]],
                        [ylower_ogrid_x0.points[::,0],ylower_ogrid_x0.points[::,1]],
                        [yupper_ogrid_x0.points[::,0],yupper_ogrid_x0.points[::,1]],
                        [ylower_ogrid_x1.points[::,0],ylower_ogrid_x1.points[::,1]]],
                       "blockboundaries")


    return 0
