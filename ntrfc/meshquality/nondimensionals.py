# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 23:33:53 2020

@author: malte
"""

import numpy as np
import pyvista as pv
from itertools import combinations

from ntrfc.math.vectorcalc import vecAbs, vecAbs_list, vecProjection, vecAngle, unitvec, unitvec_list



#################################################################################
#                                                                               #
#                                                                               #
#                          VectorHelper                                         #
#                                                                               #
#                                                                               #
#################################################################################


def cellDirections(cellUMean, wallNorm):
    x = unitvec(cellUMean)  # mainDirection
    y = unitvec(wallNorm)  # tangential
    z = unitvec(np.cross(x, y))  # spanwise
    return np.array([x, y, z])


#################################################################################
#                                                                               #
#                                                                               #
#                           CalcFunctions                                       #
#                                                                               #
#                                                                               #
#################################################################################


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

        point_pairs = list(combinations(range(len(facecenters)), 2))
        spans = [np.subtract(facecenters[pair[0]], facecenters[pair[1]]) for pair in point_pairs]
        spans_x = [vecAbs(vecProjection(xx,i)) for i in spans]
        spans_y = [vecAbs(vecProjection(yy,i)) for i in spans]
        spans_z = [vecAbs(vecProjection(zz,i)) for i in spans]
        x_span = np.max(spans_x)
        y_span = np.max(spans_y)
        z_span = np.max(spans_z)

        cell_spans.append([x_span , y_span , z_span ])
    return cell_spans



def getWalluTaus(solutionMesh, mu_0, rhofieldname, velfieldname):
    """
    Calculate the wall shear stress at various points on a surface in a CFD simulation.

    Parameters:
    solutionMesh (pv.PolyData): A mesh that represents the solution of the CFD simulation.
    surfaceMesh (pv.PolyData): A mesh that represents the surface on which the wall shear stress is to be calculated.
    mu_0 (float): The dynamic viscosity of the fluid.
    rhofieldname (str): The name of the field that contains the density of the fluid.
    velfieldname (str): The name of the field that contains the velocity of the fluid.

    Returns:
    u_taus (np.ndarray): An array containing the wall shear stress at each point on the surface.
    """


    graduwall = solutionMesh[f"grad_{velfieldname}"].reshape(solutionMesh.number_of_cells,3,3)
    wallnormals = solutionMesh["wallNormal"]
    dudy = vecAbs_list([np.dot(graduwall[i], wallnormals[i]) for i in range(solutionMesh.number_of_cells)])
    rhoWs = solutionMesh[rhofieldname]
    tauWs = dudy * mu_0
    u_taus = (tauWs / rhoWs) ** 0.5

    return u_taus


def gridSpacing(mu_0, volmesh):
    xSpans = volmesh["xSpan"]
    ySpans = volmesh["ySpan"]
    zSpans = volmesh["zSpan"]

    uTaus = volmesh["uTaus"]

    Deltax = xSpans * uTaus / mu_0
    Deltay = ySpans * uTaus / mu_0
    Deltaz = zSpans * uTaus / mu_0

    return [Deltax, Deltay, Deltaz]


def constructWallMesh(surfaces):
    wall = pv.UnstructuredGrid()
    for surf in surfaces:
        wall = wall.merge(surf)
    return wall


#################################################################################
#                                                                               #
#                                                                               #
#                    construct DimlessGridSpacing                               #
#                                                                               #
#                                                                               #
#################################################################################


def compute_scalar_gradient(mesh,arrayname):
    mesh = mesh.compute_derivative(scalars=arrayname)
    mesh[f"grad_{arrayname}"] = mesh["gradient"].reshape((mesh.number_of_cells,3,3))
    return mesh


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
    volmesh = compute_scalar_gradient(volmesh,use_velfield)
    walladjacentids = volmesh.find_containing_cell(surfacenormals_surface.points)
    volmesh_walladjacent = volmesh.extract_cells(walladjacentids)

    volmesh_walladjacent["cellCenters"] = volmesh_walladjacent.cell_centers().points
    volmesh_walladjacent["wallNormal"] = [
        surfacenormals_surface.point_data["Normals"][surfacenormals_surface.find_closest_point(i)]
        for i in volmesh_walladjacent.points]

    print("calculating cell spans from WallNormals...")
    spanS = cellSpans(volmesh_walladjacent, use_velfield)
    volmesh_walladjacent["xSpan"] = np.array([i[0] for i in spanS])  # calculate cell span in flow direction
    volmesh_walladjacent["ySpan"] = np.array([i[1] for i in spanS])  # calculate cell span in wall normal direction
    volmesh_walladjacent["zSpan"] = np.array([i[2] for i in spanS])  # calculate cell span in span direction

    print("calculating wall-shear and friction-velocity")
    volmesh_walladjacent["uTaus"] = getWalluTaus(volmesh_walladjacent, mu_0, use_rhofield, use_velfield)

    print("calculating grid spacing")
    gridSpacings = gridSpacing(mu_0, volmesh_walladjacent)

    volmesh_walladjacent["DeltaXPlus"] = gridSpacings[0]
    volmesh_walladjacent["DeltaYPlus"] = gridSpacings[1]
    volmesh_walladjacent["DeltaZPlus"] = gridSpacings[2]

    return volmesh_walladjacent

