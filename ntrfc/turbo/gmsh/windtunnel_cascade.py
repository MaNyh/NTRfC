from dataclasses import dataclass

import gmsh
import numpy as np

from ntrfc.geometry.line import lines_from_points
from ntrfc.math.vectorcalc import vecAbs
from ntrfc.turbo.cascade_case.domain import CascadeWindTunnelDomain2D
from ntrfc.turbo.cascade_case.utils.domain_utils import Blade2D, CascadeWindTunnelDomain2DParameters


@dataclass
class MeshConfig:
    """
    Configuration for meshing
    """
    lc_high: float = 0.1  # Maximum characteristic length
    lc_low: float = 0.1  # Maximum characteristic length
    bl_size: float = 0.1  # Size of boundary layer elements
    progression_le: float = 0.1
    progression_te: float = 0.1
    bl_thickness: float = 0.1  # Thickness of boundary layer


def create_mesh(domain2d: CascadeWindTunnelDomain2D,
                domainparams: CascadeWindTunnelDomain2DParameters,
                config: MeshConfig,
                blade: Blade2D,
                filename: str,
                ):
    sslengths = lines_from_points(blade.ss_pv.points).compute_cell_sizes()
    sslength = np.sum(sslengths["Length"])
    pslengths = lines_from_points(blade.ps_pv.points).compute_cell_sizes()
    pslength = np.sum(pslengths["Length"])
    ss_half_idx = np.where(np.cumsum(sslengths["Length"]) >= sslength / 2)[0][0]
    ps_half_idx = np.where(np.cumsum(pslengths["Length"]) >= pslength / 2)[0][0] + blade.ite_rolled

    path = filename
    gwkconfig = domainparams
    blades_rotated = domain2d.blades
    line0 = domain2d.line0
    line1 = domain2d.line1
    line2 = domain2d.line2
    line3 = domain2d.line3
    line4 = domain2d.line4
    line5 = domain2d.line5
    line6 = domain2d.line6
    line7 = domain2d.line7
    line8 = domain2d.line8
    line9 = domain2d.line9

    te_index = blade.ite_rolled
    sscells = int(sslength // config.lc_high)
    pscells = int(pslength // config.lc_high)

    # Initialize gmsh:
    gmsh.initialize()
    gmsh.model.add("cascade")
    blade_points_gmsh = {}

    for name, bladepoly in zip([f"blade_{i}" for i in range(domainparams.nblades)], blades_rotated):
        # Add the blade surface
        blade_points_gmsh[name] = [gmsh.model.occ.add_point(*pt, config.lc_high) for pt in bladepoly.points]

    p0 = line0.points[0]
    p1 = line1.points[0]
    p2 = line2.points[0]
    p3 = line3.points[0]
    p4 = line4.points[0]
    p5 = line5.points[0]
    p6 = line6.points[0]
    p7 = line7.points[0]
    p8 = line8.points[0]
    p9 = line9.points[0]

    p0_gmsh = gmsh.model.occ.add_point(*p0, config.lc_high)
    p1_gmsh = gmsh.model.occ.add_point(*p1, config.lc_high)
    p2_gmsh = gmsh.model.occ.add_point(*p2, config.lc_high)
    p3_gmsh = gmsh.model.occ.add_point(*p3, config.lc_high)
    p4_gmsh = gmsh.model.occ.add_point(*p4, config.lc_high)
    p5_gmsh = gmsh.model.occ.add_point(*p5, config.lc_high)
    p6_gmsh = gmsh.model.occ.add_point(*p6, config.lc_low)
    p7_gmsh = gmsh.model.occ.add_point(*p7, config.lc_low)
    p8_gmsh = gmsh.model.occ.add_point(*p8, config.lc_low)
    p9_gmsh = gmsh.model.occ.add_point(*p9, config.lc_low)

    l0_gmsh = gmsh.model.occ.addLine(p0_gmsh, p1_gmsh)
    l1_gmsh = gmsh.model.occ.addLine(p1_gmsh, p2_gmsh)
    l2_gmsh = gmsh.model.occ.addLine(p2_gmsh, p3_gmsh)
    l3_gmsh = gmsh.model.occ.addLine(p3_gmsh, p4_gmsh)
    l4_gmsh = gmsh.model.occ.addLine(p4_gmsh, p5_gmsh)
    l5_gmsh = gmsh.model.occ.addLine(p5_gmsh, p6_gmsh)
    l6_gmsh = gmsh.model.occ.addLine(p6_gmsh, p7_gmsh)
    l7_gmsh = gmsh.model.occ.addLine(p7_gmsh, p8_gmsh)
    l8_gmsh = gmsh.model.occ.addLine(p8_gmsh, p9_gmsh)
    l9_gmsh = gmsh.model.occ.addLine(p9_gmsh, p0_gmsh)
    l10_gmsh = gmsh.model.occ.addLine(p9_gmsh, p6_gmsh)

    blade_splines_gmsh = {}
    blade_curvloops_gmsh = {}

    for name, pointlist in blade_points_gmsh.items():
        blade_splines_gmsh[f"{name}_le_half_ss"] = gmsh.model.occ.add_spline(
            [*pointlist[:ss_half_idx], pointlist[ss_half_idx]])
        blade_splines_gmsh[f"{name}_halfss_te"] = gmsh.model.occ.add_spline(
            [*pointlist[ss_half_idx:te_index], pointlist[te_index]])
        blade_splines_gmsh[f"{name}_te_halfps"] = gmsh.model.occ.add_spline(
            [*pointlist[te_index:ps_half_idx], pointlist[ps_half_idx]])
        blade_splines_gmsh[f"{name}_halfps_le"] = gmsh.model.occ.add_spline(
            [*pointlist[ps_half_idx:]])#pointlist[0]
        gmsh.model.occ.synchronize()
        # Boundary layer
        f = gmsh.model.mesh.field.add('BoundaryLayer')
        gmsh.model.mesh.field.setNumbers(f, 'CurvesList',
                                         [blade_splines_gmsh[f"{name}_le_half_ss"],
                                          blade_splines_gmsh[f"{name}_halfss_te"],
                                          blade_splines_gmsh[f"{name}_te_halfps"],
                                          blade_splines_gmsh[f"{name}_halfps_le"]])
        gmsh.model.mesh.field.setNumber(f, 'Size', config.bl_size)
        gmsh.model.mesh.field.setNumber(f, 'Ratio', 1.2)
        gmsh.model.mesh.field.setNumber(f, 'Quads', 1)
        gmsh.model.mesh.field.setNumber(f, 'Thickness', config.bl_thickness)
        gmsh.model.mesh.field.setAsBoundaryLayer(f)

        gmsh.model.mesh.set_transfinite_curve(blade_splines_gmsh[f"{name}_le_half_ss"], int(sscells // 2),
                                              "Progression",
                                              1 + config.progression_le)
        gmsh.model.mesh.set_transfinite_curve(blade_splines_gmsh[f"{name}_halfss_te"], int(sscells // 2),
                                              "Progression",
                                              1 - config.progression_te)
        gmsh.model.mesh.set_transfinite_curve(blade_splines_gmsh[f"{name}_te_halfps"], int(pscells // 2),
                                              "Progression",
                                              1 + config.progression_te)
        gmsh.model.mesh.set_transfinite_curve(blade_splines_gmsh[f"{name}_halfps_le"], int(pscells // 2),
                                              "Progression",
                                              1 - config.progression_le)

        blade_curvloops_gmsh[f"{name}"] = gmsh.model.occ.add_curve_loop(
            [blade_splines_gmsh[f"{name}_le_half_ss"], blade_splines_gmsh[f"{name}_halfss_te"],
             blade_splines_gmsh[f"{name}_te_halfps"], blade_splines_gmsh[f"{name}_halfps_le"]])

    f = gmsh.model.mesh.field.add('BoundaryLayer')
    gmsh.model.mesh.field.setNumbers(f, 'CurvesList',
                                     [l0_gmsh, l1_gmsh])
    gmsh.model.mesh.field.setNumber(f, 'Size', config.bl_size)
    gmsh.model.mesh.field.setNumber(f, 'Ratio', 1.2)
    gmsh.model.mesh.field.setNumber(f, 'Quads', 1)
    gmsh.model.mesh.field.setNumber(f, 'Thickness', config.bl_thickness)
    gmsh.model.mesh.field.setNumbers(f, "PointsList", (p0_gmsh, p2_gmsh))
    gmsh.model.mesh.field.setAsBoundaryLayer(f)
    gmsh.model.occ.synchronize()

    f = gmsh.model.mesh.field.add('BoundaryLayer')
    gmsh.model.mesh.field.setNumbers(f, 'CurvesList',
                                     [l3_gmsh, l4_gmsh])
    gmsh.model.mesh.field.setNumber(f, 'Size', config.bl_size)
    gmsh.model.mesh.field.setNumber(f, 'Ratio', 1.2)
    gmsh.model.mesh.field.setNumber(f, 'Quads', 1)
    gmsh.model.mesh.field.setNumber(f, 'Thickness', config.bl_thickness)
    gmsh.model.mesh.field.setNumbers(f, "PointsList", (p3_gmsh, p5_gmsh))
    gmsh.model.mesh.field.setAsBoundaryLayer(f)
    gmsh.model.occ.synchronize()

    domain_curveloop_highres_gmsh = gmsh.model.occ.add_curve_loop(
        [l0_gmsh, l1_gmsh, l2_gmsh, l3_gmsh, l4_gmsh, l5_gmsh, l6_gmsh, l7_gmsh, l8_gmsh, l9_gmsh])

    domain_curveloop_lowres_gmsh = gmsh.model.occ.add_curve_loop([l10_gmsh, l6_gmsh, l7_gmsh, l8_gmsh])

    gmsh.model.occ.add_plane_surface(
        [domain_curveloop_highres_gmsh, *blade_curvloops_gmsh.values()])
    gmsh.model.occ.add_plane_surface([domain_curveloop_lowres_gmsh])

    l0_res = int(vecAbs(p1 - p0) / config.lc_high * 2 / 3)
    l1_res = int(vecAbs(p2 - p1) / config.lc_high * 2 / 3)
    l2_res = int(vecAbs(p3 - p2) / config.lc_high * 1 / 4)
    l3_res = int(vecAbs(p4 - p3) / config.lc_high * 2 / 3)
    l4_res = int(vecAbs(p5 - p4) / config.lc_high * 2 / 3)
    l7_res = int(vecAbs(p7 - p8) / config.lc_high * 1 / 4)

    gmsh.model.mesh.set_transfinite_curve(l2_gmsh, l2_res,
                                          "Progression", 1)
    gmsh.model.mesh.set_transfinite_curve(l0_gmsh, l0_res,
                                          "Progression", 1)
    gmsh.model.mesh.set_transfinite_curve(l4_gmsh, l4_res,
                                          "Progression", 1)
    gmsh.model.mesh.set_transfinite_curve(l1_gmsh, l1_res,
                                          "Progression", 1)
    gmsh.model.mesh.set_transfinite_curve(l3_gmsh, l3_res,
                                          "Progression", 1)

    gmsh.model.mesh.set_transfinite_curve(l7_gmsh, l7_res,
                                          "Progression", 1)
    gmsh.model.occ.synchronize()
    surfaceTags = gmsh.model.getEntities(2)

    gmsh.option.set_number("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    # mesh options
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
    gmsh.option.setNumber("General.ExpertMode", 1)
    gmsh.model.mesh.setOrder(1)
    gmsh.model.occ.synchronize()
    gmsh.model.occ.extrude(surfaceTags, dx=0, dy=0, dz=config.lc_high,
                           numElements=[1],
                           recombine=True)
    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(3, [1, 2], name="fluid")

    gmsh.model.addPhysicalGroup(2, [12, 16 + (gwkconfig.nblades - 1) * 4 + 2, 8], name="inlet")
    gmsh.model.addPhysicalGroup(2, [5], name="outlet")
    gmsh.model.addPhysicalGroup(2, [1, 2], name="z_lower")
    gmsh.model.addPhysicalGroup(2, [6], name="y_upper")
    gmsh.model.addPhysicalGroup(2, [4], name="y_lower")
    gmsh.model.addPhysicalGroup(2, [7], name="y_upper_side")
    gmsh.model.addPhysicalGroup(2, [3], name="y_lower_side")
    for i in range(gwkconfig.nblades):
        gmsh.model.addPhysicalGroup(2, [13 + i * 4, 14 + i * 4, 15 + i * 4, 16 + i * 4], name=f"blade{i}")

    gmsh.model.removeEntities([(2, 9), (2, 10), (2, 11), (2, 12)])
    gmsh.model.addPhysicalGroup(2, [16 + (gwkconfig.nblades - 1) * 4 + 1, 16 + (gwkconfig.nblades - 1) * 4 + 3],
                                name="z_upper")

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.model.occ.synchronize()
    gmsh.write(path)
    gmsh.finalize()
