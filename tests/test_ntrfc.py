#!/usr/bin/env python

"""Tests for `ntrfc` package."""

import pytest
import pyvista as pv
import numpy as np

from ntrfc import ntrfc


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
    from ntrfc.utils.filehandling.read_datafiles import yaml_dict_read

    test_file = tmpdir / "test.yaml"
    with open(test_file, "w") as handle:
        handle.write("test_key: True\n")
    assert yaml_dict_read(test_file) == {"test_key": True}


def test_yamlDictWrite(tmpdir):
    """
    tests if yaml is writing and returning a known dictionary
    """
    from ntrfc.utils.filehandling.read_datafiles import yaml_dict_read, write_yaml_dict

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
    from ntrfc.utils.filehandling.read_mesh import load_mesh

    test_file = tmpdir / "tmp.vtk"
    mesh = pv.Box()
    mesh["Density"] = np.ones(mesh.number_of_cells)
    mesh.save(test_file)
    mesh_load = load_mesh(test_file)
    assert "rho" in mesh_load.array_names


def test_surface_distance():
    from ntrfc.utils.pyvista_utils.surface import calc_dist_from_surface

    surf_one = pv.Plane()
    surf_two = pv.Plane()
    z_shift = 1.0
    surf_two.points += np.array([0, 0, z_shift])
    dist = calc_dist_from_surface(surf_one, surf_two)
    assert any(dist["distances"] == z_shift)


def test_cgnsReader():
    from ntrfc.utils.filehandling.read_mesh import cgnsReader
    # todo fill
    a = cgnsReader
    return 0


def test_vtkUnstructuredGridReader():
    from ntrfc.utils.filehandling.read_mesh import vtkUnstructuredGridReader
    # todo fill
    a = vtkUnstructuredGridReader
    return 0


def test_vtkFLUENTReader():
    from ntrfc.utils.filehandling.read_mesh import vtkFLUENTReader
    # todo fill
    a = vtkFLUENTReader
    return 0


def test_pickle_operations(tmpdir):
    from ntrfc.utils.filehandling.read_datafiles import write_pickle, read_pickle, write_pickle_protocolzero

    fname = tmpdir / "test.pkl"
    dict = {"test": 1}
    write_pickle(fname, dict)
    pklread = read_pickle(fname)
    assert dict["test"] == pklread["test"]
    write_pickle_protocolzero(fname, dict)
    pklread_zero = read_pickle(fname)
    assert dict["test"] == pklread_zero["test"]


def test_refine_spline():
    from ntrfc.utils.pyvista_utils.line import refine_spline

    coarseres = 2
    line = pv.Line(resolution=coarseres)
    fineres = 100
    fline_xx, fline_yy = refine_spline(line.points[::, 0], line.points[::, 1], fineres)
    fline = pv.lines_from_points(np.stack([fline_xx, fline_yy, np.zeros(len(fline_xx))]).T)
    assert line.length == fline.length
    assert fline.number_of_points == fineres


def test_largedistance_indices():
    from ntrfc.utils.math.vectorcalc import calc_largedistant_idx

    line = pv.Line(resolution=100)
    xx, yy = line.points[::, 0], line.points[::, 1]
    id1, id2 = calc_largedistant_idx(xx, yy)
    assert id1 == 0
    assert id2 == 100


def test_midline_from_sides():
    from ntrfc.utils.geometry.pointcloud_methods import midline_from_sides
    from ntrfc.utils.math.vectorcalc import vecAbs
    from ntrfc.database.naca_airfoil_creator import naca
    from ntrfc.utils.geometry.pointcloud_methods import extractSidePolys

    res = 240
    X, Y = naca('0009', res, finite_TE=False, half_cosine_spacing=True)
    ind_hk = 0
    ind_vk = res

    points = np.stack((X[:-1], Y[:-1], np.zeros(res * 2) - 1)).T
    poly = pv.PolyData(points)
    ssPoly, psPoly = extractSidePolys(ind_hk, ind_vk, poly)

    mids = midline_from_sides(ind_hk, ind_vk, poly.points, psPoly, ssPoly)

    length = mids.length
    testlength = vecAbs(ssPoly.points[0] - ssPoly.points[-1])

    assert length == testlength, "midline not accurate"


def test_midLength():
    """
    checks weather
    """
    from ntrfc.utils.geometry.pointcloud_methods import midLength
    import numpy as np
    import pyvista as pv

    radius = 0.5
    res = 100
    mid = int(res / 2)
    theta = np.linspace(0, 2 * np.pi, 100)

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    fake_vk = 0
    fake_hk = mid
    circle = pv.PolyData(np.stack([x, y, np.zeros(len(x))]).T)
    length = midLength(fake_vk, fake_hk, circle)
    assert np.isclose(2 * radius, length, rtol=1e-4), "length should be two times the size of the defined test-circle"


def test_extractSidePolys():
    from ntrfc.utils.geometry.pointcloud_methods import extractSidePolys
    from ntrfc.database.naca_airfoil_creator import naca

    d1, d2, d3, d4 = np.random.randint(0, 9), np.random.randint(0, 9), np.random.randint(0, 9), np.random.randint(0, 9)
    digitstring = str(d1) + str(d2) + str(d3) + str(d4)

    res = 240
    X, Y = naca(digitstring, res, finite_TE=False, half_cosine_spacing=True)
    ind_hk = 0
    ind_vk = res
    points = np.stack((X[:-1], Y[:-1], np.zeros(res * 2) - 1)).T

    poly = pv.PolyData(points)
    ssPoly, psPoly = extractSidePolys(ind_hk, ind_vk, poly)

    assert ssPoly.number_of_points == psPoly.number_of_points, "number of sidepoints are not equal"



def test_extract_geo_paras():
    from ntrfc.utils.geometry.pointcloud_methods import extract_geo_paras
    from ntrfc.database.naca_airfoil_creator import naca

    naca_code = "0009"
    angle = 20  # deg
    alpha = 1
    res = 200
    xs, ys = naca(naca_code, res, finite_TE=False, half_cosine_spacing=True)
    sortedPoly = pv.PolyData(np.stack([xs, ys, np.zeros(len(xs))]).T)
    sortedPoly.rotate_z(angle)

    points, psPoly, ssPoly, ind_vk, ind_hk, midsPoly, beta_leading, beta_trailing, camber_angle = extract_geo_paras(
        sortedPoly.points, alpha)

    assert np.isclose(beta_leading , (angle + 90)), "wrong leading edge angle"
    assert np.isclose(beta_trailing , (angle + 90)), "wrong leading edge angle"
    assert np.isclose(midsPoly.length, 1)
    assert np.isclose(camber_angle, (angle + 90))
