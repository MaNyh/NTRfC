# Import modules:
import gmsh

# Initialize gmsh:

import numpy as np
import pyvista as pv
from ntrfc.geometry.line import lines_from_points
from ntrfc.math.vectorcalc import vecAbs_list

def generate_turbocascade(sortedpoly, idx_le, idx_te, per_y_upper, per_y_lower, inletPoly, outletPoly, filename,
                          verbose=False):


    ###CONFIGURATION

    min_lc = 0.000002
    max_lc = 0.02

    lc = 5e-2  # Characteristic length for blade region


    # Initialize gmsh:
    gmsh.initialize()
    points = {}
    splines = {}
    lines = {}
    curveloops = {}
    surfaceloops = {}
    surfaces = {}


    sortedpoly_spline = lines_from_points(sortedpoly.points)
    inlet_spline = lines_from_points(inletPoly.points)
    per_y_upper_spline = lines_from_points(per_y_upper.points)
    outlet_spline = lines_from_points(outletPoly.points[::-1])
    per_y_lower_spline = lines_from_points(per_y_lower.points[::-1])

    sortedpoly_spline["ids"] = np.arange(sortedpoly_spline.number_of_points)
    inlet_spline["ids"] = np.arange(inlet_spline.number_of_points)
    per_y_upper_spline["ids"] = np.arange(per_y_upper_spline.number_of_points)
    outlet_spline["ids"] = np.arange(outlet_spline.number_of_points)
    per_y_lower_spline["ids"] = np.arange(per_y_lower_spline.number_of_points)

    if verbose:
        p = pv.Plotter()
        p.add_mesh(sortedpoly_spline, label="sortedpoly_spline")
        p.add_mesh(sortedpoly_spline.points[idx_le], label="idx_le", color="green", point_size=20)
        p.add_mesh(sortedpoly_spline.points[idx_te], label="idx_te", color="red", point_size=20)
        p.add_mesh(inlet_spline, label="inlet_spline")
        p.add_mesh(per_y_upper_spline, label="per_y_upper_spline")
        p.add_mesh(outlet_spline, label="outlet_spline")
        p.add_mesh(per_y_lower_spline, label="per_y_lower_spline")
        p.add_legend()
        p.show()

    distance_leading_edge = vecAbs_list(sortedpoly_spline.points[idx_le] - sortedpoly_spline.points)
    distance_trailing_edge = vecAbs_list(sortedpoly_spline.points[idx_te] - sortedpoly_spline.points)

    lc_blade_new = np.min(np.stack([distance_leading_edge, distance_trailing_edge]).T,axis=1)
    lc_blade_new_normalized = (lc_blade_new - min(lc_blade_new)) * (max_lc - min_lc) / (max(lc_blade_new) - min(lc_blade_new)) + min_lc

    points["blade"] = [gmsh.model.occ.add_point(*pt, lcb) for pt,lcb in zip(sortedpoly_spline.points,lc_blade_new_normalized)]
    points["per_y_upper"] = [gmsh.model.occ.add_point(*pt, lc) for pt in per_y_upper.points]
    points["per_y_lower"] = [gmsh.model.occ.add_point(*pt, lc) for pt in per_y_lower.points[::-1]]
    points["inlet"] = [points["per_y_lower"][-1], points["per_y_upper"][0]]
    points["outlet"] = [points["per_y_upper"][-1], points["per_y_lower"][0]]

    splines["blade"] = gmsh.model.occ.add_spline([*points["blade"], points["blade"][0]])
    splines["inlet"] = gmsh.model.occ.add_spline(points["inlet"])
    splines["per_y_upper"] = gmsh.model.occ.add_spline(points["per_y_upper"])
    splines["outlet"] = gmsh.model.occ.add_spline(points["outlet"])
    splines["per_y_lower"] = gmsh.model.occ.add_spline(points["per_y_lower"])

    curveloops["blade"] = gmsh.model.occ.add_curve_loop([splines["blade"]])
    curveloops["domain"] = gmsh.model.occ.add_curve_loop(
        [splines["inlet"], splines["per_y_upper"], splines["outlet"], splines["per_y_lower"]])

    #surfaces["blade"] = gmsh.model.occ.add_plane_surface([curveloops["blade"]])
    surfaces["domain"] = gmsh.model.occ.add_plane_surface([curveloops["domain"], curveloops["blade"]])
    f = gmsh.model.mesh.field.add('BoundaryLayer')
    gmsh.model.mesh.field.setNumbers(f, 'CurvesList', [curveloops["blade"]])
    gmsh.model.mesh.field.setNumber(f, 'Size', 1.7e-5)
    gmsh.model.mesh.field.setNumber(f, 'Ratio', 1.2)
    gmsh.model.mesh.field.setNumber(f, 'Quads', 1)
    gmsh.model.mesh.field.setNumber(f, 'Thickness', 0.05)
    gmsh.model.mesh.field.setAsBoundaryLayer(f)
    # Extrude the domain surface in the z-direction
    volume = gmsh.model.occ.extrude([(2, surfaces["domain"])], 0, 0, 0.6,numElements=[10],recombine=True)

    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(2, [1], 1, "The 1")
    gmsh.model.addPhysicalGroup(2, [2], 2, "The 2")
    gmsh.model.addPhysicalGroup(2, [3], 3, "The 3")
    gmsh.model.addPhysicalGroup(2, [4], 4, "The 4")
    gmsh.model.addPhysicalGroup(2, [5], 5, "The 5")
    gmsh.model.addPhysicalGroup(2, [6], 6, "The 6")
    gmsh.model.addPhysicalGroup(2, [7], 7, "The 7")
    gmsh.model.addPhysicalGroup(3, [1], 1, "The volume")
    gmsh.model.occ.synchronize()



    # Generate mesh:
    gmsh.model.mesh.generate(3)

    # Write mesh data:
    gmsh.write(filename)

    return 0
