import numpy as np
import pyvista as pv


def test_loadmesh_vtk(tmpdir):
    """
    tests if a vtk mesh can be read and Density is translated to rho
    """
    from ntrfc.filehandling.mesh import load_mesh

    test_file = tmpdir / "tmp.vtk"
    mesh = pv.Box()
    mesh["Density"] = np.ones(mesh.number_of_cells)
    mesh.save(test_file)
    mesh_load = load_mesh(test_file)
    assert "Density" in mesh_load.array_names


def test_loadmesh_cgns(tmpdir):
    from ntrfc.filehandling.mesh import load_mesh
    from pyvista.examples.downloads import download_cgns_multi
    import shutil
    download = download_cgns_multi(load=False)

    test_file = tmpdir / "tmp.cgns"
    shutil.move(download, test_file)
    mesh_load = load_mesh(test_file)
    assert "Density" in mesh_load.array_names


def test_read_vtk(tmpdir):
    from ntrfc.filehandling.mesh import read_vtk
    test_file = tmpdir / "tmp.vtk"
    # offset array.  Identifies the start of each cell in the cells array
    offset = np.array([0, 9])

    # Contains information on the points composing each cell.
    # Each cell begins with the number of points in the cell and then the points
    # composing the cell
    cells = np.array([8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, 13, 14, 15])

    # cell type array. Contains the cell type of each cell
    cell_type = np.array([12, 12])

    # in this example, each cell uses separate points
    cell1 = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]
    )

    cell2 = np.array(
        [
            [0, 0, 2],
            [1, 0, 2],
            [1, 1, 2],
            [0, 1, 2],
            [0, 0, 3],
            [1, 0, 3],
            [1, 1, 3],
            [0, 1, 3],
        ]
    )

    # points of the cell array
    points = np.vstack((cell1, cell2)).astype(float)

    # create the unstructured grid directly from the numpy arrays
    # The offset is optional and will be either calculated if not given (VTK version < 9),
    # or is not necessary anymore (VTK version >= 9)
    if pv.vtk_version_info < (9,):
        grid = pv.UnstructuredGrid(offset, cells, cell_type, points)
    else:
        grid = pv.UnstructuredGrid(cells, cell_type, points)

    # save the mesh to a vtk file
    grid.save(test_file)
    mesh = read_vtk(test_file)
    assert isinstance(mesh, (pv.UnstructuredGrid, pv.StructuredGrid))


#
# def test_read_vtm():
#     mesh = read_vtm(VTM_FILE)
#     assert isinstance(mesh, (pv.UnstructuredGrid, pv.StructuredGrid))

def test_read_cgns():
    from pyvista import examples
    from ntrfc.filehandling.mesh import read_cgns
    CGNS_FILE = examples.download_cgns_structured(load=False)
    mesh = read_cgns(CGNS_FILE)
    assert isinstance(mesh, (pv.UnstructuredGrid, pv.StructuredGrid))
