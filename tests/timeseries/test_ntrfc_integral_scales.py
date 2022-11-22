import numpy as np

from ntrfc.timeseries.integral_scales import integralscales


def test_integralscales():
    time = np.linspace(0, 100, 100000)
    freq = 1
    signal = np.sin(time * freq) + 1

    ts, ls = integralscales(signal, time)
    assert np.isclose(ts, 1, rtol=4e-3), "time scale not computed accurately"
    assert np.isclose(ls, 1, rtol=4e-3), "length scale not computed accurately"
