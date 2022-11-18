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


def test_cgnsReader():
    from ntrfc.filehandling.mesh import cgnsReader
    # todo fill
    a = cgnsReader

