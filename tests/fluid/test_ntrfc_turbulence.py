import numpy as np

from ntrfc.fluid.turbulence import calcTu, calcTkeByTu, calcTke, calcFluc, calcRey


def test_calcTu():
    # Define sample inputs and expected output
    tke = 0.1
    Uabs = 2.0
    expected_output = np.sqrt(2.0 / 3.0 * tke) / max(1e-9, Uabs)

    # Call the function and compare with expected output
    actual_output = calcTu(tke, Uabs)
    assert np.isclose(actual_output, expected_output), f"Expected {expected_output}, but got {actual_output}"


def test_calcTkeByTu():
    # Test case 1
    Tu = 0.05
    Uabs = 1.0
    expected_tke = 0.0037500000000000007
    assert np.isclose(calcTkeByTu(Tu, Uabs), expected_tke)

    # Test case 2
    Tu = 0.0
    Uabs = 1.0
    expected_tke = 0.0
    assert np.isclose(calcTkeByTu(Tu, Uabs), expected_tke)


def test_calcTke():
    u_2 = 2.0
    v_2 = 3.0
    w_2 = 4.0
    expected_tke = 4.5  # calculated as 0.5 * (u_2 + v_2 + w_2)
    assert calcTke(u_2, v_2, w_2) == expected_tke


def test_calcFluc():
    series = np.random.randn(1000)  # calculated as 0.5 * (u_2 + v_2 + w_2)
    assert np.isclose(np.mean(calcFluc(series)), 0.0)
    series = np.random.randn(1000) + 1  # calculated as 0.5 * (u_2 + v_2 + w_2)
    assert np.isclose(np.mean(calcFluc(series)), 0.0)


def test_calcRey():
    series = np.stack([np.random.randn(20000), np.random.randn(20000), np.random.randn(20000)])

    uu, vv, ww, uv, uw, vw = calcRey(series[0], series[1], series[2])
    assert np.isclose(uu, 1.0, atol=5e-2)
    assert np.isclose(vv, 1.0, atol=5e-2)
    assert np.isclose(ww, 1.0, atol=5e-2)
    assert np.isclose(uv, 0.0, atol=5e-2)
    assert np.isclose(uw, 0.0, atol=5e-2)
    assert np.isclose(vw, 0.0, atol=5e-2)
