import numpy as np
import pyvista as pv

from ntrfc.utils.math.vectorcalc import lineseg_dist


def test_absVec():
    from ntrfc.utils.math.vectorcalc import vecAbs
    a = np.array([1,1])
    assert 2**.5==vecAbs(a)
    b = np.array([7,4,4])
    assert 9 == vecAbs(b)

def test_vecDir():
    from ntrfc.utils.math.vectorcalc import vecDir, vecAbs

    b = vecDir(np.array([1,1,1]))
    assert vecAbs(b) == 1.0


def test_ellipsoidVol():
    from ntrfc.utils.math.vectorcalc import ellipsoidVol
    sigma = np.array([1,1,1])
    ellipsoid = pv.ParametricEllipsoid(*sigma)
    calcVol = ellipsoidVol(sigma)
    assert np.isclose(calcVol, ellipsoid.volume, rtol=1e-03, atol=1e-03)


def test_randomUnitVec():
    from ntrfc.utils.math.vectorcalc import randomUnitVec, vecAbs
    rvec = randomUnitVec()
    assert vecAbs(rvec)==1


def test_gradToRad():
    from ntrfc.utils.math.vectorcalc import gradToRad
    angle_grad = 180
    angle_rad = gradToRad(angle_grad)
    assert np.pi == angle_rad


def test_symToMatrix():
    from ntrfc.utils.math.vectorcalc import symToMatrix
    A = np.array([1,1,1,1,1,1])
    R = symToMatrix(A)
    assert  all(np.equal(np.ones((3,3)),R).flatten())


def test_Rx():
    from ntrfc.utils.math.vectorcalc import gradToRad, Rx
    angle = 90
    R = Rx(gradToRad(angle))
    test_vec = np.array([0,0,1])
    new_vec = np.dot(R,test_vec)
    assert all(np.isclose(new_vec,np.array([0,1,0])))


def test_Ry():
    from ntrfc.utils.math.vectorcalc import gradToRad, Ry
    angle = 90
    R = Ry(gradToRad(angle))
    test_vec = np.array([1,0,0])
    new_vec = np.dot(R,test_vec)
    assert all(np.isclose(new_vec,np.array([0,0,1])))

def test_Rz():
    from ntrfc.utils.math.vectorcalc import gradToRad, Rz
    angle = 90
    R = Rz(gradToRad(angle))
    test_vec = np.array([0,1,0])
    new_vec = np.dot(R,test_vec)
    assert all(np.isclose(new_vec,np.array([1,0,0])))


def test_lineseg():
    import pyvista as pv
    line = pv.Line()
    testpt = np.array([0,1,0])
    pt_a, pt_b = line.points[0], line.points[-1]
    assert line.length == lineseg_dist(testpt, pt_a, pt_b)
