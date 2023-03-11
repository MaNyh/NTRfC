import numpy as np
import pyvista as pv

from ntrfc.math.methods import calcAnisoMatrix, calcAnisoEigs, C_barycentric, autocorr, zero_crossings, reldiff, \
    return_intersection, is_equidistant, autocorrelate, minmax_normalize
from ntrfc.timeseries.stationarity import estimate_error_jacknife


def test_ellipsoidVol():
    import numpy as np
    import pyvista as pv
    from ntrfc.math.vectorcalc import ellipsoidVol
    sigma = np.array([1, 1, 1])
    ellipsoid = pv.ParametricEllipsoid(*sigma)
    calcVol = ellipsoidVol(sigma)
    assert np.isclose(calcVol, ellipsoid.volume, rtol=1e-03, atol=1e-03)


def test_gradToRad():
    import numpy as np
    from ntrfc.math.vectorcalc import gradToRad
    angle_grad = 180
    angle_rad = gradToRad(angle_grad)
    assert np.pi == angle_rad


def test_symToMatrix():
    import numpy as np
    from ntrfc.math.vectorcalc import symToMatrix
    A = np.array([1, 1, 1, 1, 1, 1])
    R = symToMatrix(A)
    assert all(np.equal(np.ones((3, 3)), R).flatten())


def test_Rx():
    import numpy as np
    from ntrfc.math.vectorcalc import gradToRad, Rx
    angle = 90
    R = Rx(gradToRad(angle))
    test_vec = np.array([0, 0, 1])
    new_vec = np.dot(R, test_vec)
    assert all(np.isclose(new_vec, np.array([0, 1, 0])))


def test_Ry():
    import numpy as np
    from ntrfc.math.vectorcalc import gradToRad, Ry
    angle = 90
    R = Ry(gradToRad(angle))
    test_vec = np.array([1, 0, 0])
    new_vec = np.dot(R, test_vec)
    assert all(np.isclose(new_vec, np.array([0, 0, 1])))


def test_Rz():
    import numpy as np
    from ntrfc.math.vectorcalc import gradToRad, Rz
    angle = 90
    R = Rz(gradToRad(angle))
    test_vec = np.array([0, 1, 0])
    new_vec = np.dot(R, test_vec)
    assert all(np.isclose(new_vec, np.array([1, 0, 0])))


def test_lineseg():
    import pyvista as pv
    import numpy as np
    from ntrfc.math.vectorcalc import lineseg_dist
    line = pv.Line()
    testpt = np.array([0, 1, 0])
    pt_a, pt_b = line.points[0], line.points[-1]
    assert line.length == lineseg_dist(testpt, pt_a, pt_b)


def test_findNearest():
    from ntrfc.math.vectorcalc import findNearest
    import pyvista as pv
    import numpy as np
    res = 100
    line = pv.Line(resolution=res)
    point = np.array([0, 0, 0])
    near = findNearest(line.points, point)
    assert near == int(res / 2)


def test_eulersFromRPG():
    from ntrfc.math.vectorcalc import eulersFromRPG, RotFromTwoVecs, vecAngle
    import numpy as np
    a = np.array([1, 0, 0])
    b = np.array([0, 1, 0])
    cangle = vecAngle(a, b)
    R = RotFromTwoVecs(a, b)

    angle = eulersFromRPG(R)
    assert angle[0] == cangle


def test_randomOrthMat():
    from ntrfc.math.vectorcalc import randomOrthMat
    import numpy as np
    o = randomOrthMat()
    dot_o = np.dot(o, o.T)
    assert all(np.isclose(dot_o, np.identity(3)).flatten())


def test_line_intersection():
    import numpy as np
    from ntrfc.math.vectorcalc import line_intersection

    intersect = line_intersection((-1, 0), (1, 0),
                                  (0, -1), (0, 1))
    assert all(intersect == np.array([0, 0]))


def test_largedistance_indices():
    from ntrfc.math.vectorcalc import calc_largedistant_idx

    line = pv.Line(resolution=100)
    xx, yy = line.points[::, 0], line.points[::, 1]
    id1, id2 = calc_largedistant_idx(xx, yy)
    assert id1 == 0
    assert id2 == 100


def test_calcanisomatrix():
    from ntrfc.math.vectorcalc import symToMatrix
    r = np.array([1, 1, 1, 1, 1, 1])
    rm = symToMatrix(r)
    calcAnisoMatrix(rm)
    # todo implement real test


def test_calcanisoeigs():
    from ntrfc.math.vectorcalc import symToMatrix
    r = np.array([1, 1, 1, 1, 1, 1])
    rm = symToMatrix(r)
    aniso = calcAnisoMatrix(rm)
    calcAnisoEigs(aniso)
    # todo implement real test


def test_calcbarycentric():
    from ntrfc.math.vectorcalc import symToMatrix
    r = np.array([1, 1, 1, 1, 1, 1])
    rm = symToMatrix(r)
    C_barycentric(rm)
    # todo implement real test


def test_autocorr():
    time = np.linspace(0, 200, 10000)
    signal = np.sin(time)
    corr = autocorr(signal)
    zeroc = zero_crossings(corr)

    assert np.isclose(time[zeroc[0]], np.pi / 2, rtol=0.05)


def test_reldiff():
    # Test individual float values
    assert np.isclose(reldiff(20, 10), 2 / 3)
    assert np.isclose(reldiff(0, 9), 9)
    assert np.isclose(reldiff(10, 0), 10)
    assert np.isclose(reldiff(10, 10), 0)

    # Test numpy arrays
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[1, 2, 3], [4, 5, 7]])
    assert np.allclose(reldiff(a, b), [[0, 0, 0], [0, 0, 0.15384615]])

    a = np.array([0, 0, 0])
    b = np.array([1, 2, 3])
    assert np.allclose(reldiff(a, b), [1, 2, 3])

    a = np.array([[1, 0, 3], [4, 5, 6]])
    b = np.array([[1, 2, 3], [4, 5, 7]])
    assert np.allclose(reldiff(a, b), [[0, 2, 0], [0, 0, 0.15384615]])

    # Test float and array inputs
    a = 1.0
    b = np.array([[1, 2, 3], [4, 5, 6]])
    assert np.allclose(reldiff(a, b), [[0, 2 / 3, 1], [1.2, 4 / 3, 1.42857143]])

    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = 2.0
    assert np.allclose(reldiff(a, b), [[2 / 3, 0, 0.4], [2 / 3, 0.8571428571428571, 1]])


def test_return_intersection():
    hist_1 = np.array([1, 2, 3, 4])
    hist_2 = np.array([2, 3, 2, 3])
    intersection = return_intersection(hist_1, hist_2)
    assert intersection == 0.8, f"Expected 0.8, got {intersection}"

    hist_1 = np.array([1, 1, 1, 1])
    hist_2 = np.array([2, 2, 2, 2])
    intersection = return_intersection(hist_1, hist_2)
    assert intersection == 0.5, f"Expected 0.5, got {intersection}"

    hist_1 = np.array([1, 1, 1, 1])
    hist_2 = np.array([1, 1, 1, 1])
    intersection = return_intersection(hist_1, hist_2)
    assert intersection == 1.0, f"Expected 1.0, got {intersection}"


def test_is_equidistant():
    assert is_equidistant([1, 3, 5, 7, 9]) == True
    assert is_equidistant([1, 3, 5, 8, 9]) == False
    assert is_equidistant([1]) == False
    assert is_equidistant([]) == False


def test_estimate_error():
    # Generate correlated time series using sine function
    res = 10000
    timeseries = np.sin(np.linspace(0, 8 * np.pi, res)) + np.random.normal(size=res) * 0.1
    mean_error, var_error, acorr_error = estimate_error_jacknife(timeseries, block_size=res // 4)
    assert mean_error <= 0.1
    assert var_error <= 0.1
    assert acorr_error <= 0.1

    # Generate a simple Gaussian, non self-correlated distributed timeseries with a mean of 0
    timeseries = np.random.randn(res * 6)
    mean_error, var_error, acorr_error = estimate_error_jacknife(timeseries, block_size=res // 2)
    assert mean_error <= 0.2
    assert var_error <= 0.2
    assert acorr_error <= 4


def test_autocorrelate():
    # Test with a simple sinusoidal signal
    res = 48
    signal = np.sin(np.linspace(0, 4 * np.pi, res))
    expected = res / 4
    assert np.argmin(autocorrelate(signal)) == expected

    # Test with a constant signal
    signal = np.zeros(100)
    expected = 0
    assert np.argmin(autocorrelate(signal)) == expected


def test_deterministic_timescale():
    # Test with a simple sinusoidal signal
    signal = np.sin(np.linspace(0, 4 * np.pi, 100))
    expected = 1 / 2
    assert np.isclose(deterministic_timescale(signal), expected)


def deterministic_timescale(series):
    accr = autocorrelate(series)
    zcs = np.where(np.diff(np.sign(accr)))[0]
    if len(zcs) < 2:
        return False
    tsc = (zcs[1] - zcs[0]) * 2 / len(series)
    return tsc


def test_minmax_normalize():
    # Test case 1: all values are the same
    assert min(minmax_normalize(np.array([3, 3, 3, 3]))) == 3

    # Test case 3: typical input array
    input_array = np.array([1, 2, 3, 4, 5])
    expected_output = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    assert np.allclose(minmax_normalize(input_array), expected_output)

    # Test case 4: negative values in input array
    input_array = np.array([-5, -2, 0, 2, 5])
    expected_output = np.array([0., 0.3, 0.5, 0.7, 1.])
    assert np.allclose(minmax_normalize(input_array), expected_output)

    # Test case 5: positive and negative values in input array
    input_array = np.array([-3, -1, 0, 2, 5])
    expected_output = np.array([0., 0.25, 0.375, 0.625, 1.])
    assert np.allclose(minmax_normalize(input_array), expected_output)
