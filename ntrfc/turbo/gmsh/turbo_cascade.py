# Import modules:
from dataclasses import dataclass

import gmsh
import numpy as np

from ntrfc.geometry.line import lines_from_points
from ntrfc.turbo.cascade_case.domain import CascadeDomain2D


# Initialize gmsh:


@dataclass
class MeshConfig:
    """
    Configuration for meshing
    """
    max_lc: float = 0.1  # Maximum characteristic length
    lc: float = 0.1  # Characteristic length for blade region
    bladeres: int = 300  # Number of elements in blade region
    bl_thickness: float = 0.1  # Thickness of boundary layer
    bl_growratio: float = 1.1  # Growth ratio of boundary layer
    bl_size: float = 0.1  # Size of boundary layer elements
    wake_length: float = 0.1  # Length of wake region
    wake_width: float = 0.1  # Width of wake region
    wake_lc: float = 0.1  # Characteristic length of wake region
    fake_yShiftCylinder: float = 0.1  # Shift cylinder to avoid wake region
    progression_le_halfss: float = 1.01
    progression_halfss_te: float = 0.99
    progression_te_halfps: float = 1.1
    progression_halfps_le: float = 0.9
    spansize: float = 0.1
    spanres: int = 5


def generate_turbocascade(domain2d: CascadeDomain2D,
                          meshconfig: MeshConfig,
                          filename: str):
    """
    Generate a mesh for a turbocascade
    :param domain2d: Domain2D object
    :param meshconfig: MeshConfig object
    :param filename: Filename for mesh
    :param verbose: Print gmsh output
    :return:
    """

    # Initialize gmsh:
    gmsh.initialize()
    gmsh.model.add("cascade")
    points = {}
    splines = {}
    curveloops = {}
    surfaces = {}

    # Set meshing options:
    profilepoints_rolled_le = domain2d.blade.sortedpoints_pv_rolled

    te_index = domain2d.blade.ite_rolled

    profilepoints_line = lines_from_points(profilepoints_rolled_le.points).compute_cell_sizes()

    lc_blade = np.sum(profilepoints_line["Length"]) / meshconfig.bladeres

    sslengths = lines_from_points(profilepoints_line.points[:te_index]).compute_cell_sizes()
    sslength = np.sum(sslengths["Length"])
    pslengths = lines_from_points(profilepoints_line.points[te_index:]).compute_cell_sizes()
    pslength = np.sum(pslengths["Length"])

    ss_half_idx = np.where(np.cumsum(sslengths["Length"]) >= sslength / 2)[0][0]
    ps_half_idx = np.where(np.cumsum(pslengths["Length"]) >= pslength / 2)[0][0] + te_index

    # domain2d.profilepoints_spline = lines_from_points(domain2d.profilepoints.points)
    inlet_spline = lines_from_points(domain2d.inlet.points)
    domain2d.yperiodic_high_spline = lines_from_points(domain2d.yperiodic_high.points)
    outlet_spline = lines_from_points(domain2d.outlet.points[::-1])
    domain2d.yperiodic_low_spline = lines_from_points(domain2d.yperiodic_low.points[::-1])

    # domain2d.profilepoints_spline["ids"] = np.arange(domain2d.profilepoints_spline.number_of_points)
    inlet_spline["ids"] = np.arange(inlet_spline.number_of_points)
    domain2d.yperiodic_high_spline["ids"] = np.arange(domain2d.yperiodic_high_spline.number_of_points)
    outlet_spline["ids"] = np.arange(outlet_spline.number_of_points)
    domain2d.yperiodic_low_spline["ids"] = np.arange(domain2d.yperiodic_low_spline.number_of_points)

    # Create points and splines:

    points["blade"] = [gmsh.model.occ.add_point(*pt, lc_blade) for pt in profilepoints_rolled_le.points]
    points["domain2d.yperiodic_high"] = [gmsh.model.occ.add_point(*pt, meshconfig.lc) for pt in
                                         domain2d.yperiodic_high.points]
    points["domain2d.yperiodic_low"] = [gmsh.model.occ.add_point(*pt, meshconfig.lc) for pt in
                                        domain2d.yperiodic_low.points[::-1]]
    points["inlet"] = [points["domain2d.yperiodic_low"][-1], points["domain2d.yperiodic_high"][0]]
    points["outlet"] = [points["domain2d.yperiodic_high"][-1], points["domain2d.yperiodic_low"][0]]

    #    splines["blade"] = gmsh.model.occ.add_spline([*points["blade"], points["blade"][0]])
    splines["le_halfss"] = gmsh.model.occ.add_spline([*points["blade"][:ss_half_idx], points["blade"][ss_half_idx]])
    splines["halfss_te"] = gmsh.model.occ.add_spline(
        [*points["blade"][ss_half_idx:te_index], points["blade"][te_index]])
    splines["te_halfps"] = gmsh.model.occ.add_spline(
        [*points["blade"][te_index:ps_half_idx], points["blade"][ps_half_idx]])
    splines["halfps_le"] = gmsh.model.occ.add_spline([*points["blade"][ps_half_idx:], points["blade"][0]])

    splines["inlet"] = gmsh.model.occ.add_spline(points["inlet"])
    splines["domain2d.yperiodic_high"] = gmsh.model.occ.add_spline(points["domain2d.yperiodic_high"])
    splines["outlet"] = gmsh.model.occ.add_spline(points["outlet"])
    splines["domain2d.yperiodic_low"] = gmsh.model.occ.add_spline(points["domain2d.yperiodic_low"])

    curveloops["blade"] = gmsh.model.occ.add_curve_loop(
        [splines["le_halfss"], splines["halfss_te"], splines["te_halfps"], splines["halfps_le"]])
    curveloops["domain"] = gmsh.model.occ.add_curve_loop(
        [splines["inlet"], splines["domain2d.yperiodic_high"], splines["outlet"], splines["domain2d.yperiodic_low"]])

    surfaces["domain"] = gmsh.model.occ.add_plane_surface([curveloops["domain"], curveloops["blade"]])
    gmsh.model.occ.synchronize()

    # Boundary layer
    f = gmsh.model.mesh.field.add('BoundaryLayer')
    gmsh.model.mesh.field.setNumbers(f, 'CurvesList', [splines["le_halfss"], splines["halfss_te"], splines["te_halfps"],
                                                       splines["halfps_le"]])
    gmsh.model.mesh.field.setNumber(f, 'Size', meshconfig.bl_size)
    gmsh.model.mesh.field.setNumber(f, 'Ratio', meshconfig.bl_growratio)
    gmsh.model.mesh.field.setNumber(f, 'Quads', 1)
    gmsh.model.mesh.field.setNumber(f, 'Thickness', meshconfig.bl_thickness)
    gmsh.model.mesh.field.setAsBoundaryLayer(f)

    # blade resolution

    curvelength = sslength + pslength

    sscells = meshconfig.bladeres * sslength / curvelength
    pscells = meshconfig.bladeres * pslength / curvelength

    gmsh.model.mesh.set_transfinite_curve(splines["le_halfss"], int(sscells // 2), "Progression",
                                          meshconfig.progression_le_halfss)
    gmsh.model.mesh.set_transfinite_curve(splines["halfss_te"], int(sscells // 2), "Progression",
                                          meshconfig.progression_halfss_te)
    gmsh.model.mesh.set_transfinite_curve(splines["te_halfps"], int(pscells // 2), "Progression",
                                          meshconfig.progression_te_halfps)
    gmsh.model.mesh.set_transfinite_curve(splines["halfps_le"], int(pscells // 2), "Progression",
                                          meshconfig.progression_halfps_le)

    # Wake Resolution
    w = gmsh.model.mesh.field.add('Cylinder')
    gmsh.model.mesh.field.setNumber(w, "VIn", meshconfig.wake_lc)
    gmsh.model.mesh.field.setNumber(w, "VOut", meshconfig.max_lc)
    gmsh.model.mesh.field.setNumber(w, "Radius", meshconfig.wake_width)
    gmsh.model.mesh.field.setNumber(w, "XAxis", 0.5 * meshconfig.wake_length)

    minx = np.min(profilepoints_rolled_le.points[::, 0])
    maxx = np.max(profilepoints_rolled_le.points[::, 0])
    miny = np.min(profilepoints_rolled_le.points[::, 1])
    # maxy = np.max(profilepoints_rolled_le.points[::, 1])

    wake_angle = np.deg2rad(-domain2d.blade.beta_te)

    gmsh.model.mesh.field.setNumber(w, "XCenter", minx + (maxx - minx) + 0.5 * meshconfig.wake_length)
    gmsh.model.mesh.field.setNumber(w, "YCenter",
                                    meshconfig.fake_yShiftCylinder + miny - 0.5 * meshconfig.wake_length * np.tan(
                                        wake_angle))
    gmsh.model.mesh.field.setNumber(w, "ZAxis", 0)
    gmsh.model.mesh.field.setNumber(w, "YAxis", -0.5 * meshconfig.wake_length * np.tan(wake_angle))
    gmsh.model.mesh.field.setNumber(w, "XAxis", 0.5 * meshconfig.wake_length)
    gmsh.model.mesh.field.setAsBackgroundMesh(w)
    gmsh.model.occ.synchronize()
    surfaceTags = gmsh.model.getEntities(2)

    gmsh.option.set_number("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    # mesh options
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
    # gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)  # all hexahedra

    gmsh.model.mesh.setOrder(1)
    gmsh.model.occ.synchronize()
    gmsh.model.occ.extrude(surfaceTags, dx=0, dy=0, dz=meshconfig.spansize,
                           numElements=[meshconfig.spanres],
                           recombine=True)
    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(3, [1], name="fluid")
    gmsh.model.addPhysicalGroup(2, [2], name="inlet")
    gmsh.model.addPhysicalGroup(2, [4], name="outlet")
    gmsh.model.addPhysicalGroup(2, [6, 7, 8, 9], name="blade")
    gmsh.model.addPhysicalGroup(2, [1], name="z_lower")
    gmsh.model.addPhysicalGroup(2, [10], name="z_upper")
    gmsh.model.addPhysicalGroup(2, [3], name="y_upper")
    gmsh.model.addPhysicalGroup(2, [5], name="y_lower")

    gmsh.model.occ.synchronize()

    gmsh.model.mesh.generate(3)
    gmsh.model.occ.synchronize()

    gmsh.write(filename)
    gmsh.model.removePhysicalGroups(gmsh.model.occ.getEntities(0))
    gmsh.model.removePhysicalGroups(gmsh.model.occ.getEntities(1))
    gmsh.finalize()
