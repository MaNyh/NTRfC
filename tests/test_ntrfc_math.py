
def test_absVec():
    import numpy as np
    from ntrfc.utils.math.vectorcalc import vecAbs
    a = np.array([1, 1])
    assert 2 ** .5 == vecAbs(a)
    b = np.array([7, 4, 4])
    assert 9 == vecAbs(b)


def test_vecDir():
    import numpy as np
    from ntrfc.utils.math.vectorcalc import vecDir, vecAbs

    b = vecDir(np.array([1, 1, 1]))
    assert vecAbs(b) == 1.0


def test_ellipsoidVol():
    import numpy as np
    import pyvista as pv
    from ntrfc.utils.math.vectorcalc import ellipsoidVol
    sigma = np.array([1, 1, 1])
    ellipsoid = pv.ParametricEllipsoid(*sigma)
    calcVol = ellipsoidVol(sigma)
    assert np.isclose(calcVol, ellipsoid.volume, rtol=1e-03, atol=1e-03)


def test_randomUnitVec():
    import numpy as np
    from ntrfc.utils.math.vectorcalc import randomUnitVec, vecAbs
    rvec = randomUnitVec()
    assert np.isclose(vecAbs(rvec), 1)


def test_gradToRad():
    import numpy as np
    from ntrfc.utils.math.vectorcalc import gradToRad
    angle_grad = 180
    angle_rad = gradToRad(angle_grad)
    assert np.pi == angle_rad


def test_symToMatrix():
    import numpy as np
    from ntrfc.utils.math.vectorcalc import symToMatrix
    A = np.array([1, 1, 1, 1, 1, 1])
    R = symToMatrix(A)
    assert all(np.equal(np.ones((3, 3)), R).flatten())


def test_Rx():
    import numpy as np
    from ntrfc.utils.math.vectorcalc import gradToRad, Rx
    angle = 90
    R = Rx(gradToRad(angle))
    test_vec = np.array([0, 0, 1])
    new_vec = np.dot(R, test_vec)
    assert all(np.isclose(new_vec, np.array([0, 1, 0])))


def test_Ry():
    import numpy as np
    from ntrfc.utils.math.vectorcalc import gradToRad, Ry
    angle = 90
    R = Ry(gradToRad(angle))
    test_vec = np.array([1, 0, 0])
    new_vec = np.dot(R, test_vec)
    assert all(np.isclose(new_vec, np.array([0, 0, 1])))


def test_Rz():
    import numpy as np
    from ntrfc.utils.math.vectorcalc import gradToRad, Rz
    angle = 90
    R = Rz(gradToRad(angle))
    test_vec = np.array([0, 1, 0])
    new_vec = np.dot(R, test_vec)
    assert all(np.isclose(new_vec, np.array([1, 0, 0])))


def test_lineseg():
    import pyvista as pv
    import numpy as np
    from ntrfc.utils.math.vectorcalc import lineseg_dist
    line = pv.Line()
    testpt = np.array([0, 1, 0])
    pt_a, pt_b = line.points[0], line.points[-1]
    assert line.length == lineseg_dist(testpt, pt_a, pt_b)


def test_RotFromTwoVecs():
    import numpy as np
    from ntrfc.utils.math.vectorcalc import Rz
    from ntrfc.utils.math.vectorcalc import RotFromTwoVecs
    a = np.array([1, 0, 0])
    b = np.array([0, 1, 0])

    Rab = RotFromTwoVecs(b,a)
    Rcontrol = Rz(np.pi/2)
    assert all(np.isclose(Rab,Rcontrol).flatten())

def test_radiusFromPt():
    from ntrfc.utils.math.vectorcalc import radiusFromPt
    a = radiusFromPt


def test_posVec():
    import numpy as np
    from ntrfc.utils.math.vectorcalc import vecAbs, posVec

    a = np.array([-1, 0, 0])
    alength = vecAbs(a)
    b = posVec(a)
    blength = vecAbs(b)
    assert alength==blength
    assert np.equal(-1*a,b)


def test_findNearest():
    from ntrfc.utils.math.vectorcalc import findNearest
    a = findNearest


def test_eulersFromRPG():
    from ntrfc.utils.math.vectorcalc import eulersFromRPG
    a = eulersFromRPG


def test_randomOrthMat():
    from ntrfc.utils.math.vectorcalc import randomOrthMat
    a = randomOrthMat


def test_minDists():
    from ntrfc.utils.math.vectorcalc import minDists
    a = minDists


def test_vecProjection():
    import numpy as np
    from ntrfc.utils.math.vectorcalc import vecProjection, vecAbs

    a = np.array([1, 0, 0])
    b = np.array([0, 1, 0])
    c = vecProjection(a, b)

    assert vecAbs(c) == 0.0

    d = np.array([1, 0, 0])
    e = np.array([1, 1, 0])
    f = vecProjection(d, e)
    assert vecAbs(f) == 1.0


def test_vecAngle():
    import numpy as np
    from ntrfc.utils.math.vectorcalc import vecAngle
    a = np.array([1, 0, 0])
    b = np.array([0, 1, 0])
    angle = vecAngle(a, b)
    assert angle == np.pi / 2
