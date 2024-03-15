import numpy as np
import pyvista as pv

from ntrfc.geometry.line import polyline_from_points, refine_spline
from ntrfc.math.vectorcalc import vecAngle
from ntrfc.turbo.cascade_geometry import calcmidpassagestreamline


def create_profileprobes(ssPoly, psPoly, midspan_z, pden_ps, pden_ss, tolerance=1e-10):
    """
    Create profile probes from two PolyData objects.

    Parameters:
    - ssPoly: PyVista PolyData object representing the suction side profile.
    - psPoly: PyVista PolyData object representing the pressure side profile.
    - midspan_z: Height of the midspan plane along the z-axis.
    - pden_ps: Density of the pressure side profile points.
    - pden_ss: Density of the suction side profile points.
    - tolerance: Small tolerance value to shift the 3D faces along their normals.

    Returns:
    - probes_ss: PyVista PolyData object representing the profile probes on the suction side.
    - probes_ps: PyVista PolyData object representing the profile probes on the pressure side.
    """

    # Refine the splines defined by the input PolyData objects
    ref_ss_x, ref_ss_y = refine_spline(ssPoly.points[::, 0], ssPoly.points[::, 1], 4000)
    ref_ps_x, ref_ps_y = refine_spline(psPoly.points[::, 0], psPoly.points[::, 1], 4000)

    # Create PolyData objects from the refined splines
    ref_ss_points = np.stack((ref_ss_x, ref_ss_y, np.zeros(len(ref_ss_y)))).T
    ref_ps_points = np.stack((ref_ps_x, ref_ps_y, np.zeros(len(ref_ps_y)))).T
    ref_ssPoly = pv.PolyData(ref_ss_points)
    ref_psPoly = pv.PolyData(ref_ps_points)

    # Convert the PolyData objects to polylines
    ref_ss_poly = polyline_from_points(ref_ssPoly.points)
    ref_ps_poly = polyline_from_points(ref_psPoly.points)

    # Extrude the polylines along the z-axis to create 3D face models
    ref_ss_face = ref_ss_poly.extrude((0, 0, midspan_z * 2), capping=False).compute_normals()
    ref_ps_face = ref_ps_poly.extrude((0, 0, midspan_z * 2), capping=False).compute_normals()

    # Shift the 3D faces slightly along their normals
    ref_ss_face_shift = ref_ss_face.copy()
    ref_ss_face_shift.points += tolerance * ref_ss_face_shift.point_data["Normals"]
    ref_ps_face_shift = ref_ps_face.copy()
    ref_ps_face_shift.points += tolerance * ref_ps_face_shift.point_data["Normals"]

    # Create a cut through each 3D face at the midspan plane
    ref_ss_cut = ref_ss_face_shift.slice(normal="z", origin=(0, 0, midspan_z))
    ref_ps_cut = ref_ps_face_shift.slice(normal="z", origin=(0, 0, midspan_z))

    # Extract the x and y coordinates of the points on the cut faces
    x_ss_shift = ref_ss_cut.points[::, 0]
    y_ss_shift = ref_ss_cut.points[::, 1]
    x_ps_shift = ref_ps_cut.points[::, 0]
    y_ps_shift = ref_ps_cut.points[::, 1]

    # Refine the splines defined by the cut faces using the specified densities
    x_bl_ss, y_bl_ss = refine_spline(x_ss_shift, y_ss_shift, pden_ss)
    x_bl_ps, y_bl_ps = refine_spline(x_ps_shift, y_ps_shift, pden_ps)

    # Create PolyData objects from the refined splines
    probes_ps = pv.PolyData(np.stack((x_bl_ss, y_bl_ss, midspan_z * np.ones(len(x_bl_ss)))).T)
    probes_ss = pv.PolyData(np.stack((x_bl_ps, y_bl_ps, midspan_z * np.ones(len(x_bl_ps)))).T)

    # Return the profile probes
    return probes_ss, probes_ps


def create_midpassageprobes(midspan_z, x_inlet, x_outlet, pitch, beta1, beta2, midspoly, nop_streamline):
    x_mcl, y_mcl = midspoly.points[::, 0], midspoly.points[::, 1]

    x_mpsl, y_mpsl = calcmidpassagestreamline(x_mcl, y_mcl, beta1, beta2, x_inlet, x_outlet, pitch)

    # x_probes = []
    # y_probes = []
    z_probes = []

    nop = int(nop_streamline)

    xn, yn = refine_spline(x_mpsl, y_mpsl, nop)
    for i in range(nop):
        z_probes.append(midspan_z)

    x_probes = xn
    y_probes = yn

    dist = np.sqrt((x_probes[0] - x_probes[1]) ** 2 + (y_probes[0] - y_probes[1]) ** 2)

    x_probes[0] = x_probes[0] + 0.00001 * dist
    x_probes[-1] = x_probes[-1] - 0.00001 * dist

    return pv.PolyData(np.stack([x_probes, y_probes, np.ones(len(x_probes) * midspan_z)]).T)


def create_stagnationpointprobes(length, nop, sortedpoly, ind_vk, u_inlet, midspan_z):
    vk_point = sortedpoly.points[ind_vk]
    angle = vecAngle(u_inlet, np.array([1, 0, 0]) * 180 / np.pi)
    stagnationLine = pv.Line((0, 0, 0), (-length, 0, 0), nop - 1)
    stagnationLine.rotate_z(angle, inplace=False)
    stagnationLine.translate(vk_point, inplace=False)
    x_probes = stagnationLine.points[::, 0]
    y_probes = stagnationLine.points[::, 1]
    z_probes = stagnationLine.points[::, 2] + midspan_z
    return pv.PolyData(np.stack([x_probes, y_probes, z_probes]).T)
