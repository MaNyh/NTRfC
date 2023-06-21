# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 23:33:53 2020

@author: malte
"""

import numpy as np
import pyvista as pv
import vtk
from scipy.spatial import KDTree

from ntrfc.math.vectorcalc import vecAbs_list, unitvec


def cellDirections(cellUMean, wallNorm):
    x = unitvec(cellUMean)  # mainDirection
    y = unitvec(wallNorm)  # tangential
    z = unitvec(np.cross(x, y))  # spanwise
    return np.array([x, y, z])


def cellSpans(solutionMesh, calcFrom):
    cell_spans = []
    cellIds = np.arange(0, solutionMesh.number_of_cells)
    for cellIdx in cellIds:
        wallNormal = solutionMesh["wallNormal"][cellIdx]
        uMean = solutionMesh[calcFrom][cellIdx]

        cellDirs = cellDirections(uMean, wallNormal)
        xx = cellDirs[0]
        yy = cellDirs[1]
        zz = cellDirs[2]
        cellfaces = solutionMesh.extract_cells(cellIdx).extract_surface()
        facecenters = cellfaces.cell_centers().points
        spans = facecenters[:, np.newaxis] - facecenters
        spans_x = np.abs(np.dot(spans, xx))
        spans_y = np.abs(np.dot(spans, yy))
        spans_z = np.abs(np.dot(spans, zz))
        x_span, y_span, z_span = np.max(spans_x), np.max(spans_y), np.max(spans_z)

        cell_spans.append([x_span, y_span, z_span])
    return cell_spans


def get_wall_shear_stress_velocity(mesh, dynamic_viscosity, density_fieldname, velocity_fieldname):
    """
    Calculate the wall shear stress velocity at various points on a surface in a CFD simulation.

    Parameters:
    mesh (pv.PolyData): A mesh that represents the solution_utils of the CFD simulation.
    dynamic_viscosity (float): The dynamic viscosity of the fluid.
    density_fieldname (str): The name of the field that contains the density of the fluid.
    velocity_fieldname (str): The name of the field that contains the velocity of the fluid.

    Returns:
    wall_shear_stress_velocity (np.ndarray): An array containing the velocity at which a fluid layer adjacent to
    the surface would need to move in order to experience the same shear stress as the actual fluid layer in contact
    with the surface.
    """

    grad_velocity = mesh[f"grad_{velocity_fieldname}"].reshape(mesh.number_of_cells, 3, 3)
    wall_normals = mesh["wallNormal"]
    velocity_gradient_normal = vecAbs_list(
        [np.dot(grad_velocity[i], wall_normals[i]) for i in range(mesh.number_of_cells)])
    fluid_density = mesh[density_fieldname]
    tau_w = dynamic_viscosity / fluid_density * velocity_gradient_normal
    return np.sqrt(tau_w / fluid_density)


def constructWallMesh(surfaces):
    wall = pv.UnstructuredGrid()
    for surf in surfaces:
        wall = wall.merge(surf)
    return wall


def compute_scalar_gradient(mesh, arrayname):
    mesh = mesh.compute_derivative(arrayname)
    mesh[f"grad_{arrayname}"] = mesh["gradient"]
    return mesh


def calc_dimensionless_yplus(volmesh, surfaces, use_velfield, use_rhofield, mu_0):
    surfaceMesh = constructWallMesh(surfaces)
    surfacenormals_surface = surfaceMesh.extract_surface().compute_normals()

    volmesh = compute_scalar_gradient(volmesh, use_velfield)
    walladjacentids = volmesh.find_containing_cell(surfacenormals_surface.cell_centers().points)
    volmesh_walladjacent = volmesh.extract_cells(walladjacentids)
    volmesh_walladjacent["cellCenters"] = volmesh_walladjacent.cell_centers().points
    volmesh_walladjacent["wallNormal"] = [
        surfacenormals_surface.point_data["Normals"][surfacenormals_surface.find_closest_point(i)]
        for i in volmesh_walladjacent.points]
    distcompute = pv.PolyData(volmesh_walladjacent["cellCenters"]).compute_implicit_distance(
        surfaceMesh.extract_surface())
    volmesh_walladjacent["cellCentersWallDistance"] = distcompute["implicit_distance"]
    volmesh_walladjacent["uTaus"] = get_wall_shear_stress_velocity(volmesh_walladjacent, mu_0, use_rhofield,
                                                                   use_velfield)
    YPlus = np.abs(volmesh_walladjacent["cellCentersWallDistance"]) * volmesh_walladjacent["uTaus"] / mu_0
    volmesh_walladjacent["yPlus"] = YPlus
    return volmesh_walladjacent


def calc_dimensionless_gridspacing(volmesh, surfaces, use_velfield, use_rhofield, mu_0):
    """
    :param volmesh: pyvista-vtk object
    :param surfaces: pyvista-vtk object
    :param use_velfield: string, name of the velocity field array
    :param use_rhofield:  string, name of the density field array
    :param mu_0: float. kinematic viscosity
    :return: volmesh_walladjacent: pyvista-vtk object with the nondimensionals
    """

    print("constructing surfacemesh from wall meshes ...")
    surfaceMesh = constructWallMesh(surfaces)
    print("calculating wall-normal vectors...")
    surfacenormals_surface = surfaceMesh.extract_surface().compute_normals()

    print("finding walladjacent cells")
    merged_mesh = surfacenormals_surface + volmesh
    # Compute cell-based derivative of a vector field
    derivatives = compute_scalar_gradient(merged_mesh, "U")

    derivatives = derivatives.ctp()
    # Extract the volumetric mesh of the derivative-field
    cell_types = merged_mesh.celltypes
    volumetric_cells = np.where(cell_types == vtk.VTK_HEXAHEDRON)[0]
    face_cells = np.where(cell_types == vtk.VTK_QUAD)[0]
    # Extract the derivative-field
    volmesh = derivatives.extract_cells(volumetric_cells)
    facemesh = derivatives.extract_cells(face_cells).ctp()
    walladjacentids = volmesh.find_containing_cell(surfacenormals_surface.cell_centers().points)
    volmesh_walladjacent = volmesh.extract_cells(walladjacentids)
    volmesh_walladjacent["cellCenters"] = volmesh_walladjacent.cell_centers().points

    facemesh = facemesh.extract_surface().compute_normals()
    # Extract the cell centers of the wall-adjacent cells and the surface cells
    volmesh_walladjacent_centers = volmesh_walladjacent.cell_centers().points
    surfacemesh_surface_centers = facemesh.cell_centers().points
    facemesh = facemesh.ptc()
    # Construct a KDTree from the surface cell centers and their normals
    surface_normals = facemesh["Normals"]
    surface_gradients = facemesh["grad_U"]
    tree = KDTree(surfacemesh_surface_centers)

    # Find the indices of the closest surface cell centers to each wall-adjacent cell center
    distances, indices = tree.query(volmesh_walladjacent_centers)

    # Assign the corresponding surface normals to the wall-adjacent cells
    volmesh_walladjacent["wallNormal"] = surface_normals[indices]
    volmesh_walladjacent["wallUGradient"] = surface_gradients[indices]

    print("calculating cell spans from WallNormals...")
    spanS = cellSpans(volmesh_walladjacent, use_velfield)
    volmesh_walladjacent["xSpan"] = np.array([i[0] for i in spanS])  # calculate cell span in flow direction
    volmesh_walladjacent["ySpan"] = np.array([i[1] for i in spanS])  # calculate cell span in wall normal direction
    volmesh_walladjacent["zSpan"] = np.array([i[2] for i in spanS])  # calculate cell span in span direction

    print("calculating wall-shear and friction-velocity")
    volmesh_walladjacent = volmesh_walladjacent.ptc()
    grad_velocity = volmesh_walladjacent["wallUGradient"].reshape(volmesh_walladjacent.number_of_cells, 3, 3)
    wall_normals = volmesh_walladjacent["wallNormal"]
    velocity_gradient_normal = vecAbs_list(
        [np.dot(grad_velocity[i], wall_normals[i]) for i in range(volmesh_walladjacent.number_of_cells)])
    fluid_density = volmesh_walladjacent[use_rhofield]
    u_tau = np.sqrt(mu_0 * velocity_gradient_normal / fluid_density)
    # uTaus = volmesh_walladjacent["uTaus"]
    d_v = mu_0 / volmesh_walladjacent[use_rhofield] / u_tau
    volmesh_walladjacent["DeltaXPlus"] = volmesh_walladjacent[
                                             "xSpan"] / d_v  # calculate cell span in flow direction
    volmesh_walladjacent["DeltaYPlus"] = volmesh_walladjacent[
                                             "ySpan"] / d_v  # calculate cell span in wall normal direction
    volmesh_walladjacent["DeltaZPlus"] = volmesh_walladjacent[
                                             "zSpan"] / d_v  # calculate cell span in span direction

    return volmesh_walladjacent
