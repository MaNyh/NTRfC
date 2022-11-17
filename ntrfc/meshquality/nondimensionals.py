# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 23:33:53 2020

@author: malte
"""

import numpy as np
import pyvista as pv

from ntrfc.math.vectorcalc import vecAbs,vecAbs_list, vecProjection, vecAngle, unitvec, unitvec_list


def readDataSet(grid, dataName):
    grid.set_active_scalars(dataName)
    data = grid.active_scalars
    return data


#################################################################################
#                                                                               #
#                                                                               #
#                          VectorHelper                                         #
#                                                                               #
#                                                                               #
#################################################################################


def cellDirections(cellUMean, wallNorm):
    x = unitvec(cellUMean)  # mainDirection
    z = unitvec(wallNorm)  # tangential
    y = unitvec(np.cross(x, z))  # spanwise
    return np.array([x, y, z])


#################################################################################
#                                                                               #
#                                                                               #
#                           CalcFunctions                                       #
#                                                                               #
#                                                                               #
#################################################################################


def closestWallNormalPoint(point, surfacenormals):
    surfacepoint_id = surfacenormals.find_closest_point(point)
    wallpoint = surfacenormals.points[surfacepoint_id]
    return surfacenormals.point_arrays["Normals"][surfacepoint_id], wallpoint - point


def calcWallNormalVectors(surfaceMesh, volmesh):
    surfacenormals = []
    surfacevectors = []
    surfacenormals_surface = surfaceMesh.extract_surface().compute_normals()
    cellIds = np.arange(0,volmesh.number_of_cells)
    for cellIdx in cellIds:
        center = volmesh["cellCenters"][cellIdx]
        surfacenormal, surfacevector = closestWallNormalPoint(center, surfacenormals_surface)
        surfacenormals.append(surfacenormal)
        surfacevectors.append(surfacevector)

    surfacenormals = np.array(surfacenormals)
    surfacevectors = np.array(surfacevectors)
    return surfacenormals, surfacevectors


def cellSpans( solutionMesh, calcFrom):
    spans = []
    cellIds = np.arange(0,solutionMesh.number_of_cells)
    for cellIdx in cellIds:

        x_span = 0
        x_weight = 0

        y_span = 0
        y_weight = 0

        z_span = 0
        z_weight = 0

        wallNormal = solutionMesh["wallNormal"][cellIdx]
        uMean = solutionMesh[calcFrom][cellIdx]

        cellDirs = cellDirections(uMean, wallNormal)
        xx = cellDirs[0]
        yy = cellDirs[1]
        zz = cellDirs[2]

        egdeVectors = []

        CELL = solutionMesh.GetCell(cellIdx)
        # CELL.points=np.dot(np.array(cellDirs),CELL.points)
        edgeNumbers = CELL.GetNumberOfEdges()

        for edgeIdx in range(edgeNumbers):
            EDGE = CELL.GetEdge(edgeIdx)
            EDGE = EDGE.GetPoints()
            edgeVec = np.array(EDGE.GetPoint(0)) - np.array(EDGE.GetPoint(1))
            egdeVectors.append(edgeVec)

        for vec in egdeVectors:
            x_span += vecAbs(vecProjection(xx, vec))
            x_weight += abs(np.cos(vecAngle(xx, vec)))

            y_span += vecAbs(vecProjection(yy, vec))
            y_weight += abs(np.cos(vecAngle(yy, vec)))

            z_span += vecAbs(vecProjection(zz, vec))
            z_weight += abs(np.cos(vecAngle(zz, vec)))

        spans.append([x_span / x_weight, y_span / y_weight, z_span / z_weight])
        # pBarUpdate()
    return spans


def getWalluTaus(labelChunk, solutionMesh, surfaceMesh, mu_0, rhofieldname, velfieldname):
    uTaus = []
    surfaceNormals = surfaceMesh.extract_surface().compute_normals()
    cell_centers = solutionMesh["cellCenters"]

    relatedpointonwallids = [surfaceNormals.find_closest_point(i) for i in cell_centers]
    relatedpointonwall = [surfaceNormals.points[i] for i in relatedpointonwallids]
    relatedpointonwall_samples = pv.PolyData(relatedpointonwall).sample(surfaceMesh)
    graduwall = relatedpointonwall_samples["gradient"].reshape(relatedpointonwall_samples.number_of_points,3,3)
    flowdirections = unitvec_list(solutionMesh[velfieldname])
    gradUyWs = vecAbs_list([np.dot(flowdirections[i], graduwall[i]) for i in range(relatedpointonwall_samples.number_of_points)])
    rhoWs = relatedpointonwall_samples[rhofieldname]

    tauWs = gradUyWs * mu_0 * rhoWs
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

def calc_dimensionless_gridspacing(volmesh, surfaces, use_velfield, use_rhofield, mu_0):
    """
    :param volmesh: pyvista-vtk object
    :param surfaces: pyvista-vtk object
    :param use_velfield: string, name of the velocity field array
    :param use_rhofield:  string, name of the density field array
    :param mu_0: float. kinematic viscosity
    :return: volmesh: pyvista-vtk object with the nondimensionals
    """

    print("constructing surfacemesh from wall meshes ...")
    surfaceMesh = constructWallMesh(surfaces)
    surfaceMeshcopy = surfaceMesh.copy()
    surfaceMeshcopy.clear_data()

    print("preparing processData from meshes")

    volmesh = volmesh.compute_derivative(scalars=use_velfield)
    surfaceMeshcopy = surfaceMeshcopy.sample(volmesh)
    surfaceMesh["gradient"] = surfaceMeshcopy["gradient"]
    volmesh[use_velfield] = readDataSet(volmesh, use_velfield)
    volmesh["cellCenters"] = volmesh.cell_centers().points

    cellIds = [i for i in range(volmesh.GetNumberOfCells())]

    print("calculating wall-normal vectors...")
    surfaceNormals, surfaceVectors = calcWallNormalVectors(surfaceMesh, volmesh)
    volmesh["wallNormal"] = surfaceNormals

    print("calculating cell spans from WallNormals and CellEdges...")
    spanS = cellSpans( volmesh, use_velfield)
    volmesh["xSpan"] = np.array([i[0] for i in spanS])  # calculate cell span in flow direction
    volmesh["ySpan"] = np.array([i[1] for i in spanS])  # calculate cell span in wall normal direction
    volmesh["zSpan"] = np.array([i[2] for i in spanS])  # calculate cell span in span direction

    print("calculating wall-shear and friction-velocity")
    uTaus = getWalluTaus(cellIds, volmesh, surfaceMesh, mu_0, use_rhofield, use_velfield)
    volmesh["uTaus"] = uTaus

    print("calculating grid spacing")
    gridSpacings = gridSpacing(mu_0, volmesh)

    volmesh["DeltaXPlus"] = gridSpacings[0]
    volmesh["DeltaYPlus"] = gridSpacings[1]
    volmesh["DeltaZPlus"] = gridSpacings[2]

    return volmesh
