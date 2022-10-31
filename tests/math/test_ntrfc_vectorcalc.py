def test_absVec():
    import numpy as np
    from ntrfc.math.vectorcalc import vecAbs
    a = np.array([1, 1])
    assert 2 ** .5 == vecAbs(a)
    b = np.array([7, 4, 4])
    assert 9 == vecAbs(b)


def test_vecDir():
    import numpy as np
    from ntrfc.math.vectorcalc import vecDir, vecAbs

    b = vecDir(np.array([1, 1, 1]))
    assert vecAbs(b) == 1.0


def test_randomUnitVec():
    import numpy as np
    from ntrfc.math.vectorcalc import randomUnitVec, vecAbs
    rvec = randomUnitVec()
    assert np.isclose(vecAbs(rvec), 1)


def test_RotFromTwoVecs():
    import numpy as np
    from ntrfc.math.vectorcalc import Rz
    from ntrfc.math.vectorcalc import RotFromTwoVecs
    a = np.array([1, 0, 0])
    b = np.array([0, 1, 0])

    Rab = RotFromTwoVecs(b, a)
    Rcontrol = Rz(np.pi / 2)
    assert all(np.isclose(Rab, Rcontrol).flatten())


def test_posVec():
    import numpy as np
    from ntrfc.math.vectorcalc import vecAbs, posVec

    a = np.array([-1, 0, 0])
    alength = vecAbs(a)
    b = posVec(a)
    blength = vecAbs(b)
    assert alength == blength
    assert all(np.isclose(-1 * a, b).flatten())


def test_vecProjection():
    import numpy as np
    from ntrfc.math.vectorcalc import vecProjection, vecAbs

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
    from ntrfc.math.vectorcalc import vecAngle
    a = np.array([1, 0, 0])
    b = np.array([0, 1, 0])
    angle = vecAngle(a, b)
    assert angle == np.pi / 2
