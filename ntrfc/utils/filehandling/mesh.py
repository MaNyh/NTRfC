import os
import vtk
import pyvista as pv


def load_mesh(path_to_mesh):
    assert os.path.isfile(path_to_mesh), path_to_mesh + " is not a valid file"
    extension = os.path.splitext(path_to_mesh)[1]
    if extension == ".vtk" or extension == ".vtp" or extension == ".vtu" :
        mesh = pv.read(path_to_mesh)
    elif  extension == ".vtm":
        multiBlockMesh = pv.read(cgns)
        mesh = multiBlockMesh.combine()
    elif extension == ".cgns":
        cgns = cgnsReader(path_to_mesh)
        if str(type(cgns)).find('vtkStructuredGrid') != -1:
            mesh = pv.StructuredGrid(cgns)
        elif str(type(cgns)).find('vtkUnstructuredGrid') != -1:
            mesh = pv.UnstructuredGrid(cgns)
        elif str(type(cgns)).find('vtkMultiBlockDataSet') != -1:
            multiBlockMesh = pv.MultiBlock(cgns)
            mesh = multiBlockMesh.combine()
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
