import numpy as np
import pyvista as pv



def test_absVec():
    from ntrfc.utils.math.vectorcalc import vecAbs
    a = np.array([1,1])
    assert np.sqrt(2)==vecAbs(a)
    b = np.array([1,1,1])
    assert np.sqrt(3) == vecAbs(b)

def test_unitVec():
    from ntrfc.utils.math.vectorcalc import unitVec

    b = unitVec(np.array([1,1,1]))
    assert absVec(b) == 1.0


def test_ellipsoidVol():
    from ntrfc.utils.math.vectorcalc import ellipsoidVol
    sigma = np.array([1,1,1])
    ellipsoid = pv.ParametricEllipsoid(*sigma)
    calcVol = ellipsoidVol(sigma)
    assert np.isclose(calcVol, ellipsoid.volume, rtol=1e-03, atol=1e-03)
