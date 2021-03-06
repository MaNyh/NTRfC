#!/usr/bin/env python

"""Tests for `ntrfc` package."""
import os

import pytest
import pyvista as pv
import numpy as np



@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_yamlDictRead(tmpdir):
    """
    tests if yaml is returning a known dictionary
    """
    from ntrfc.utils.filehandling.datafiles import yaml_dict_read

    test_file = tmpdir / "test.yaml"
    with open(test_file, "w") as handle:
        handle.write("test_key: True\n")
    assert yaml_dict_read(test_file) == {"test_key": True}


def test_yamlDictWrite(tmpdir):
    """
    tests if yaml is writing and returning a known dictionary
    """
    from ntrfc.utils.filehandling.datafiles import yaml_dict_read, write_yaml_dict

    test_file = tmpdir / "test.yaml"
    test_dict = {"test_key": True}
    write_yaml_dict(test_file, test_dict)

    assert yaml_dict_read(test_file) == test_dict


def test_polyline_from_points():
    from ntrfc.utils.pyvista_utils.line import polyline_from_points

    points = pv.Line(resolution=100).points
    line = polyline_from_points(points)
    assert line.length == 1.0, "theres something fishy about the polyline_from_points implementation"


def test_line_from_points():
    from ntrfc.utils.pyvista_utils.line import lines_from_points

    points = pv.Line(resolution=100).points
    line = lines_from_points(points)
    assert line.length == 1.0, "theres something fishy about the polyline_from_points implementation"


def test_loadmesh_vtk(tmpdir):
    """
    tests if a vtk mesh can be read and Density is translated to rho
    """
    from ntrfc.utils.filehandling.mesh import load_mesh

    test_file = tmpdir / "tmp.vtk"
    mesh = pv.Box()
    mesh["Density"] = np.ones(mesh.number_of_cells)
    mesh.save(test_file)
    mesh_load = load_mesh(test_file)
    assert "Density" in mesh_load.array_names


def test_surface_distance():
    from ntrfc.utils.pyvista_utils.surface import calc_dist_from_surface

    surf_one = pv.Plane()
    surf_two = pv.Plane()
    z_shift = 1.0
    surf_two.points += np.array([0, 0, z_shift])
    dist = calc_dist_from_surface(surf_one, surf_two)
    assert any(dist["distances"] == z_shift)


def test_cgnsReader():
    from ntrfc.utils.filehandling.mesh import cgnsReader
    # todo fill
    a = cgnsReader
    return 0


def test_vtkUnstructuredGridReader():
    from ntrfc.utils.filehandling.mesh import vtkUnstructuredGridReader
    # todo fill
    a = vtkUnstructuredGridReader
    return 0


def test_vtkFLUENTReader():
    from ntrfc.utils.filehandling.mesh import vtkFLUENTReader
    # todo fill
    a = vtkFLUENTReader
    return 0


def test_pickle_operations(tmpdir):
    """
    checks if the pickle-operators are working
    :param tmpdir:
    :return:
    """
    from ntrfc.utils.filehandling.datafiles import write_pickle, read_pickle, write_pickle_protocolzero

    fname = tmpdir / "test.pkl"
    dict = {"test": 1}
    write_pickle(fname, dict)
    pklread = read_pickle(fname)
    assert dict["test"] == pklread["test"]
    write_pickle_protocolzero(fname, dict)
    pklread_zero = read_pickle(fname)
    assert dict["test"] == pklread_zero["test"]


def test_refine_spline():
    """
    tests if you can refine a spline by checking the number of points and the length of the spline
    """
    from ntrfc.utils.pyvista_utils.line import refine_spline

    coarseres = 2
    line = pv.Line(resolution=coarseres)
    fineres = 100
    fline_xx, fline_yy = refine_spline(line.points[::, 0], line.points[::, 1], fineres)
    fline = pv.lines_from_points(np.stack([fline_xx, fline_yy, np.zeros(len(fline_xx))]).T)
    assert line.length == fline.length
    assert fline.number_of_points == fineres


def test_create_dirstructure(tmpdir):
    from ntrfc.utils.filehandling.datafiles import create_dirstructure

    dirstructure = ["ast/bla", "ast/ble"]
    create_dirstructure(dirstructure, tmpdir)
    checks = [os.path.isdir(path) for path in [os.path.join(tmpdir, relpath) for relpath in dirstructure]]
    assert all(checks), "not all directories have been created"


