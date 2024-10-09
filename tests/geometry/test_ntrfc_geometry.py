import pytest


def test_extract_vk_hk(verbose=False):
    """
    tests a NACA  profile in a random angle as a minimal example.
    :return:
    """
    from ntrfc.turbo.profile_tele_extraction import extract_vk_hk
    from ntrfc.turbo.airfoil_generators.naca_airfoil_creator import naca
    from ntrfc.geometry.alphashape import auto_concaveHull
    import numpy as np
    import pyvista as pv

    # d1,d2,d3,d4 = np.random.randint(0,9),np.random.randint(0,9),np.random.randint(0,9),np.random.randint(0,9)
    # digitstring = str(d1)+str(d2)+str(d3)+str(d4)
    # manifold problems with other profiles with veronoi-mid and other unoptimized code. therefor tests only 0009
    # todo: currently we cant test half_cosine_spacing profiles, as the resolution is too good for extract_vk_hk
    naca_code = "6509"
    angle = 0  # deg
    res = 512
    xs, ys = naca(naca_code, res, half_cosine_spacing=False, finite_te=False)
    xs_n, ys_n,_ = auto_concaveHull(xs,ys)
    sorted_poly = pv.PolyData(np.stack([xs_n, ys_n, np.zeros(len(xs_n))]).T)
    sorted_poly.rotate_z(angle, inplace=True)

    ind_vk, ind_hk = extract_vk_hk(sorted_poly)

    if verbose:
        p = pv.Plotter()
        p.add_mesh(sorted_poly.points[ind_hk], color="yellow", point_size=20)
        p.add_mesh(sorted_poly.points[ind_vk], color="red", point_size=20)
        p.add_mesh(sorted_poly)
        p.view_xy()
        p.show()

    assert all(sorted_poly.points[np.argmax(sorted_poly.points[:,0])] == sorted_poly.points[ind_hk])
    assert all(sorted_poly.points[np.argmin(sorted_poly.points[:,0])] == sorted_poly.points[ind_vk])






def test_midline_from_sides(verbose=False):
    from ntrfc.turbo.cascade_case.utils.domain_utils import Blade2D
    from ntrfc.math.vectorcalc import vecAbs
    from ntrfc.turbo.airfoil_generators.naca_airfoil_creator import naca
    import numpy as np
    import pyvista as pv

    res = 256
    x, y = naca('0009', res, half_cosine_spacing=True)

    points = np.stack((x[:], y[:], np.zeros(len(x)))).T
    blade = Blade2D(points)
    blade.compute_all_frompoints()

    length = blade.skeletonline_pv.length
    testlength = vecAbs(blade.ss_pv.points[0] - blade.ss_pv.points[-1])

    if verbose:
        p = pv.Plotter()
        p.add_mesh(blade.sortedpoints_pv)
        p.add_mesh(blade.skeletonline_pv)
        p.add_mesh(blade.sortedpoints_pv.points[blade.ite], color="k", label="hk", point_size=20)
        p.add_mesh(blade.sortedpoints_pv.points[blade.ile], color="g", label="vk", point_size=20)
        p.add_legend()
        p.show()

    assert np.isclose(length, testlength, rtol=1e-3), "midline not accurate"


@pytest.mark.parametrize("xs, ys, expected_result", [
    ([0, 1, 2, 2, 1, 0], [0, 0, 1, 2, 2, 1], True),
    ([0, 1, 2, 2, 1, 0], [0, 1, 2, 2, 1, 0], False),
])
def test_orientation_of_circle(xs, ys, expected_result):
    from ntrfc.turbo.pointcloud_methods import is_counterclockwise
    assert is_counterclockwise(xs, ys) is expected_result


def test_inside_poly():
    from ntrfc.geometry.plane import inside_poly
    # Test for a simple polygon and point
    polygon = [(0, 0), (0, 1), (1, 1), (1, 0)]
    point = [(0.5, 0.5)]
    assert inside_poly(polygon, point)[0] == True

    # Test for a point outside the polygon
    polygon = [(0, 0), (0, 1), (1, 1), (1, 0)]
    point = [(1.5, 1.5)]
    assert inside_poly(polygon, point)[0] == False

    # Test for a point on the boundary of the polygon
    polygon = [(0, 0), (0, 1), (1, 1), (1, 0)]
    point = [(1, 1)]
    assert inside_poly(polygon, point)[0] == False
