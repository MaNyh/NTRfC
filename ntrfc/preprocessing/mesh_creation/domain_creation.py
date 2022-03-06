from ntrfc.utils.geometry.pointcloud_methods import calcMidPassageStreamLine
from ntrfc.utils.pyvista_utils.line import lines_from_points
from ntrfc.utils.math.vectorcalc import closest_node_index, vecAbs, vecDir
from ntrfc.utils.filehandling.tecplot import writeTecplot1DFile
from ntrfc.utils.filehandling.datafiles import write_pickle
from ntrfc.utils.filehandling.datafiles import read_pickle

import numpy as np
import pyvista as pv
import os
import matplotlib.pyplot as plt


def create_2d_domain(basedir, profile_set, geometry):
    for profile_name in profile_set:
        pitch = geometry["pitch"]
        x_inlet = geometry["x_inlet"]
        x_outlet = geometry["x_outlet"]
        blade_shift = geometry["blade_shift"]

        profile_pickle = os.path.join(basedir, "01_profile", profile_name + "_profile.pkl")

        profile_dict = read_pickle(profile_pickle)
        beta_meta_01 = profile_dict["beta_meta_01"]
        beta_meta_02 = profile_dict["beta_meta_02"]

        midsPoly = profile_dict["midsPoly"]
        ssPoly = profile_dict["ssPoly"]
        psPoly = profile_dict["psPoly"]

        x_mids = midsPoly[::, 0]
        y_mids = midsPoly[::, 1]
        x_ss = ssPoly[::, 0]
        y_ss = ssPoly[::, 1]
        x_ps = psPoly[::, 0]
        y_ps = psPoly[::, 1]

        # stagger_angle = np.rad2deg(np.arctan((y_mids[-1] - y_mids[-0]) / (x_mids[-1] - x_mids[-0])))

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

        geometry_paras = {
            "inlet": inletPoly.points,
            "outlet": outletPoly.points,
            "per_y_upper": per_y_upper.points,
            "per_y_lower": per_y_lower.points, }

        domain_dir = os.path.join(basedir, "02_domain")
        write_pickle(os.path.join(domain_dir, profile_name + "_domain.pkl"), geometry_paras)

        plt.figure()
        plt.plot(inletPoly.points[::, 0], inletPoly.points[::, 1], color="#6c3376")
        plt.plot(outletPoly.points[::, 0], outletPoly.points[::, 1], color="#FF2211")
        plt.plot(per_y_upper.points[::, 0], per_y_upper.points[::, 1], color="#FF22CC")
        plt.plot(per_y_lower.points[::, 0], per_y_lower.points[::, 1], color="#FF22CC")
        plt.plot(x_ss, y_ss, color="#FFAA44")
        plt.plot(x_ps, y_ps, color="#FFAA44")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(os.path.join(domain_dir, profile_name + '_domain.pdf'))


def create_2d_blocklines(basedir, profile_set):
    for profile_name in profile_set:
        profile_dict = read_pickle(os.path.join(basedir, "01_profile", profile_name + "_profile.pkl"))
        domain_dict = read_pickle(os.path.join(basedir, "02_domain", profile_name + "_domain.pkl"))

        sortedPoints = profile_dict["sortedPoints"]
        midsPoly = profile_dict["midsPoly"]
        bladeline = lines_from_points(sortedPoints)
        y_upper = domain_dict["per_y_upper"]
        y_lower = domain_dict["per_y_lower"]
        inlet = domain_dict["inlet"]
        outlet = domain_dict["outlet"]

        ssPoly = profile_dict["ssPoly"]
        psPoly = profile_dict["psPoly"]

        bladehelpersurface = bladeline.extrude((0, 0, 0.01))

        ogridhelpersurface = bladehelpersurface.compute_normals()

        # compute size of ogrid. there are probably better solutions
        ogrid_zero = bladeline.points[0]
        upper_idx = closest_node_index(ogrid_zero, y_upper)
        lower_idx = closest_node_index(ogrid_zero, y_lower)
        distance_upper = vecAbs(y_upper[upper_idx] - ogrid_zero)
        distance_lower = vecAbs(y_lower[lower_idx] - ogrid_zero)
        ogrid_size = 0.3 * min([distance_upper, distance_lower])

        chord_start = 0.01
        chord_idx_helper = int(chord_start * len(midsPoly))
        cstart_1 = midsPoly[chord_idx_helper]
        cstart_2 = midsPoly[-chord_idx_helper]

        ogridhelpersurface.points += ogrid_size * ogridhelpersurface.point_normals
        ogridline = ogridhelpersurface.slice(normal="z")
        ogridline.points[:, 2] = 0
        domain_dir = os.path.join(basedir, "03_meshgeometry")

        for chord_start in range(len(midsPoly)):
            ss_le = closest_node_index(midsPoly[chord_start], ssPoly)
            ps_le = closest_node_index(midsPoly[chord_start], psPoly)

            # ss_le_ogrid = closest_node_index(ss_le, ogridline.points)
            # ps_le_ogrid = closest_node_index(ps_le, ogridline.points)

            # yupper_le = closest_node_index(ss_le_ogrid, y_upper)
            # ylower_le = closest_node_index(ps_le_ogrid, y_lower)

            a = 1
            # angle_

        ps_x0_blade = closest_node_index(cstart_1, psPoly)
        ps_x1_blade = closest_node_index(cstart_2, psPoly)

        ss_x0_blade = closest_node_index(cstart_1, ssPoly)
        ss_x1_blade = closest_node_index(cstart_2, ssPoly)

        ps_x0_ogrid_idx = closest_node_index(psPoly[ps_x0_blade], ogridline.points)
        ps_x1_ogrid_idx = closest_node_index(psPoly[ps_x1_blade], ogridline.points)

        ss_x0_ogrid_idx = closest_node_index(ssPoly[ss_x0_blade], ogridline.points)
        ss_x1_ogrid_idx = closest_node_index(ssPoly[ss_x1_blade], ogridline.points)

        ps_x0_ogrid_line = pv.Line(ogridline.points[ps_x0_ogrid_idx], psPoly[ps_x0_blade])
        ps_x1_ogrid_line = pv.Line(ogridline.points[ps_x1_ogrid_idx], psPoly[ps_x1_blade])

        ss_x0_ogrid_line = pv.Line(ogridline.points[ss_x0_ogrid_idx], ssPoly[ss_x0_blade])
        ss_x1_ogrid_line = pv.Line(ogridline.points[ss_x1_ogrid_idx], ssPoly[ss_x1_blade])

        yperlow_x0_ps_idx = np.argmin((cstart_1[0] - y_lower[::, 0]) ** 2)
        yperlow_x1_ps_idx = np.argmin((cstart_2[0] - y_lower[::, 0]) ** 2)

        ylower_ogrid_x0 = pv.Line(y_lower[yperlow_x0_ps_idx], ogridline.points[ss_x0_ogrid_idx])
        yupper_ogrid_x0 = pv.Line(y_upper[yperlow_x0_ps_idx], ogridline.points[ps_x0_ogrid_idx])

        ylower_ogrid_x1 = pv.Line(y_lower[yperlow_x1_ps_idx], ogridline.points[ss_x1_ogrid_idx])
        yupper_ogrid_x1 = pv.Line(y_upper[yperlow_x1_ps_idx], ogridline.points[ps_x1_ogrid_idx])

        msp_xx, msp_yy = calcMidPassageStreamLine(midsPoly[::, 0], midsPoly[::, 1],
                                                  profile_dict["beta_meta_01"],
                                                  profile_dict["beta_meta_02"],
                                                  pv.PolyData(inlet).bounds[0], pv.PolyData(outlet).bounds[0], 0)

        mspPoly = lines_from_points(np.stack([msp_xx, msp_yy, np.zeros(len(msp_yy))]).T)

        le_ogrid = pv.Line(midsPoly[0], midsPoly[0] - vecDir(midsPoly[1] - midsPoly[0]) * ogrid_size)
        te_ogrid = pv.Line(midsPoly[-1], midsPoly[-1] - vecDir(midsPoly[-2] - midsPoly[-1]) * ogrid_size)
        ogrid_inlet_dist = vecAbs(mspPoly.points[0] - le_ogrid.points[-1])
        ogrid_outlet_dist = vecAbs(mspPoly.points[-1] - te_ogrid.points[-1])
        ogrid_inlet = pv.Line(le_ogrid.points[-1],
                              le_ogrid.points[-1] - vecDir(midsPoly[1] - midsPoly[0]) * ogrid_inlet_dist)
        ogrid_oulet = pv.Line(te_ogrid.points[-1],
                              te_ogrid.points[-1] - vecDir(midsPoly[-2] - midsPoly[-1]) * ogrid_outlet_dist)

        plt.figure()
        plt.plot(inlet[::, 0], inlet[::, 1], color="#6c3376")
        plt.plot(outlet[::, 0], outlet[::, 1], color="#FF2211")
        plt.plot(y_upper[::, 0], y_upper[::, 1], color="#FF22CC")
        plt.plot(y_lower[::, 0], y_lower[::, 1], color="#FF22CC")
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
        plt.savefig(os.path.join(domain_dir, profile_name + '_blocklines.pdf'))

        writeTecplot1DFile(os.path.join(domain_dir, profile_name + "_domain.geom"), ['x', 'z'],
                           ["inlet", "outlet", "y_upper", "y_lower", "psPoly", "ssPoly"],
                           [[inlet[::, 0], inlet[::, 1]],
                            [outlet[::, 0], outlet[::, 1]],
                            [y_upper[::, 0], y_upper[::, 1]],
                            [y_lower[::, 0], y_lower[::, 1]],
                            [psPoly[::, 0], psPoly[::, 1]],
                            [ssPoly[::, 0], ssPoly[::, 1]]],
                           profile_name + "Domainboundaries")

        writeTecplot1DFile(os.path.join(domain_dir, profile_name + "_blocks.geom"), ['x', 'z'],
                           ["le_ogrid",
                            "te_ogrid",
                            "ogrid_inlet",
                            "ogrid_oulet",
                            "ogridline",
                            "ps_x0_ogrid_line",
                            "ps_x1_ogrid_line",
                            "ss_x0_ogrid_line",
                            "ss_x1_ogrid_line",
                            "ylower_ogrid_x0",
                            "yupper_ogrid_x0",
                            "ylower_ogrid_x1",
                            "yupper_ogrid_x1"],
                           [[le_ogrid.points[::, 0], le_ogrid.points[::, 1]],
                            [te_ogrid.points[::, 0], te_ogrid.points[::, 1]],
                            [ogrid_inlet.points[::, 0], ogrid_inlet.points[::, 1]],
                            [ogrid_oulet.points[::, 0], ogrid_oulet.points[::, 1]],
                            [ogridline.points[::, 0], ogridline.points[::, 1]],
                            [ps_x0_ogrid_line.points[::, 0], ps_x0_ogrid_line.points[::, 1]],
                            [ps_x1_ogrid_line.points[::, 0], ps_x1_ogrid_line.points[::, 1]],
                            [ss_x0_ogrid_line.points[::, 0], ss_x0_ogrid_line.points[::, 1]],
                            [ss_x1_ogrid_line.points[::, 0], ss_x1_ogrid_line.points[::, 1]],
                            [ylower_ogrid_x0.points[::, 0], ylower_ogrid_x0.points[::, 1]],
                            [yupper_ogrid_x0.points[::, 0], yupper_ogrid_x0.points[::, 1]],
                            [ylower_ogrid_x1.points[::, 0], ylower_ogrid_x1.points[::, 1]]],
                           profile_name + "Blockboundaries")
