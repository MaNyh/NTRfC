import os
import vtk
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
