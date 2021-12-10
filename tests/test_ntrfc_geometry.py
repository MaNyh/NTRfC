import numpy as np
import pyvista as pv



def test_calcConcaveHull():
    """
    in these simple geometries, each point must be found by calcConcaveHull
    """
    from ntrfc.utils.geometry.pointcloud_methods import calcConcaveHull

    square = pv.Plane()
    boxedges = square.extract_feature_edges()

    boxedges.rotate_z(np.random.randint(0, 360))
    boxpoints = boxedges.points

    np.random.shuffle(boxpoints)

    xs_raw = boxpoints[:, 0]
    ys_raw = boxpoints[:, 1]

    xs, ys = calcConcaveHull(xs_raw, ys_raw, 10)

    assert len(xs) == len(xs_raw)
    assert any([yi in ys_raw for yi in ys])

    polygon = pv.Polygon()
    polygon.rotate_z(np.random.randint(0, 360))
    polyedges = polygon.extract_feature_edges()
    polypoints = polyedges.points
    np.random.shuffle(polypoints)
    xs_raw = polypoints[:, 0]
    ys_raw = polypoints[:, 1]

    xs, ys = calcConcaveHull(xs_raw, ys_raw, 10)

    assert len(xs) == len(xs_raw)
    assert any([yi in ys_raw for yi in ys])


def test_parsec():
    from ntrfc.database.parsec_airfoil_creator import parsec_airfoil_gen

    R_LE = 0.01
    x_PRE = 0.450
    y_PRE = -0.006
    d2y_dx2_PRE = -0.4
    th_PRE = 0.05
    x_SUC = 0.450
    y_SUC = 0.055
    d2y_dx2_SUC = -0.350
    th_SUC = -6

    pparray = [R_LE,x_PRE,y_PRE,d2y_dx2_PRE,th_PRE,x_SUC,y_SUC,d2y_dx2_SUC,th_SUC]
    profile_points = parsec_airfoil_gen(pparray)
    X,Y = profile_points[::,0],profile_points[::,1]


def test_naca():
    def rand_naca_code():
        digits = np.random.choice([4,5])
        if digits ==4:
            d1,d2,d3,d4 = np.random.randint(0,6),np.random.randint(0,8),np.random.randint(0,6),np.random.randint(0,9)
            digitstring = str(d1)+str(d2)+str(d3)+str(d4)
        if digits ==5:
            d1,d2,d3,d4,d5 = np.random.randint(0,5),np.random.randint(0,5),np.random.randint(0,6),np.random.randint(0,9),np.random.randint(0,9)
            digitstring = str(d1)+str(d2)+str(d3)+str(d4)+str(d5)
        return digitstring
    profNaca = [rand_naca_code() for i in range(6)]
