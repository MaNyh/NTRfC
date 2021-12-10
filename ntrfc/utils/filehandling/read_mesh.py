import os
import vtk
import numpy as np
import pyvista as pv


def load_mesh(path_to_mesh):
    assert os.path.isfile(path_to_mesh), path_to_mesh + " is not a valid file"
    extension = os.path.splitext(path_to_mesh)[1]
    if extension == ".vtk":
        mesh = pv.read(path_to_mesh)

    elif extension == ".cgns":
        cgns = cgnsReader(path_to_mesh)
        if str(type(cgns)).find('vtkStructuredGrid') != -1:
            mesh = pv.StructuredGrid(cgns)
        elif str(type(cgns)).find('vtkUnstructuredGrid') != -1:
            mesh = pv.UnstructuredGrid(cgns)
        elif str(type(cgns)).find('vtkMultiBlockDataSet') != -1:
            mesh = pv.UnstructuredGrid()
            multiBlockMesh = pv.MultiBlock(cgns)
            for domainId in range(multiBlockMesh.GetNumberOfBlocks()):
                domain = multiBlockMesh.GetBlock(domainId)
                for blockId in range(domain.GetNumberOfBlocks()):
                    block = domain.GetBlock(blockId)
                    mesh = mesh.merge(block)

    array_names = mesh.array_names
    for name in array_names:
        if name == "Velocity":
            mesh.rename_array("Velocity", "U")
        if name == "Pressure":
            mesh.rename_array("Pressure", "p")
        if name == "Density":
            mesh.rename_array("Density", "rho")
        if name == "Temperature":
            mesh.rename_array("Temperature", "T")
    mesh = mesh.cell_data_to_point_data()

    return mesh


def polyline_from_points(points):
    poly = pv.PolyData()
    poly.points = points
    the_cell = np.arange(0, len(points), dtype=np.int_)
    the_cell = np.insert(the_cell, 0, len(points))
    poly.lines = the_cell
    return poly


def lines_from_points(points):
    """Given an array of points, make a line set"""
    poly = pv.PolyData()
    poly.points = points
    cells = np.full((len(points) - 1, 3), 2, dtype=np.int_)
    cells[:, 1] = np.arange(0, len(points) - 1, dtype=np.int_)
    cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
    poly.lines = cells
    return poly


def calc_dist_from_surface(surface_primary, surface_secondary, verbose=False):
    """
    Distance Between Two Surfaces / A Surface and a Pointcloud
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Compute the average thickness between two surfaces.

    We can compute the thickness between the two surfaces using a few different
    methods. We will demo a method where we compute the normals of the
    bottom surface, and then project a ray to the top surface to compute the
    distance along the surface normals.
    :param surface_primary: pv.UnstructuredGrid
    :param surface_secondary: pv.UnstructuredGrid / pv.PolyData
    :param verbose: plots?
    :return: surface_primary with distances from secondary
    """

    ###############################################################################
    # Ray Tracing Distance
    # ++++++++++++++++++++
    #

    h0n = surface_primary.compute_normals(point_normals=True, cell_normals=False,
                                          auto_orient_normals=True)

    ###############################################################################
    # Travel along normals to the other surface and compute the thickness on each
    # vector.

    h0n["distances"] = np.empty(surface_primary.n_points)
    for i in range(h0n.n_points):
        p = h0n.points[i]
        vec = h0n["Normals"][i] * h0n.length
        p0 = p - vec
        p1 = p + vec
        ip, ic = surface_secondary.ray_trace(p0, p1, first_point=True)
        dist = np.sqrt(np.sum((ip - p) ** 2))
        h0n["distances"][i] = dist

    if any(h0n["distances"] == 0):
        # Replace zeros with nans
        mask = h0n["distances"] == 0
        h0n["distances"][mask] = np.nan
        np.nanmean(h0n["distances"])

    if verbose:
        ###############################################################################
        p = pv.Plotter()
        p.add_mesh(h0n, scalars="distances", smooth_shading=True)
        p.add_mesh(surface_secondary, color=True, opacity=0.75, smooth_shading=True)
        p.show()

    return h0n


def cgnsReader(file):
    reader = vtk.vtkCGNSReader()
    reader.SetFileName(file)
    reader.UpdateInformation()
    reader.EnableAllCellArrays()
    reader.EnableAllPointArrays()
    reader.Update()
    dataset = reader.GetOutput()
    return dataset

def vtkUnstructuredGridReader(file):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(file)
    reader.UpdateInformation()
    reader.Update()
    dataset = reader.GetOutput()
    return dataset

def vtkFLUENTReader(file):
    reader = vtk.vtkFLUENTReader
    reader.SetFileName(file)
    reader.UpdateInformation()
    reader.Update()
    dataset = reader.GetOutput()
    return dataset
