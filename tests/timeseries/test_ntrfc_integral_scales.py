import numpy as np

from ntrfc.timeseries.integral_scales import integralscales, get_self_correlating_frequencies


def test_integralscales():
    time = np.linspace(0, 100, 100000)
    freq = 1
    signal = np.sin(time * freq) + 1

    ts, ls = integralscales(signal, time)
    assert np.isclose(ts, 1, rtol=4e-3), "time scale not computed accurately"
    assert np.isclose(ls, 1, rtol=4e-3), "length scale not computed accurately"


def test_get_self_correlating_frequencies():
    # Test a sinusoidal signal with a frequency of 1Hz
    timesteps = np.linspace(0, 10, 1000)
    signal = np.sin(2 * np.pi * timesteps)
    frequencies = get_self_correlating_frequencies(timesteps, signal)
    assert np.isclose(frequencies[0], 1.0, rtol=1e-2)

    # Test a sinusoidal signal with a frequency of 2Hz
    timesteps = np.linspace(0, 5, 500)
    signal = np.sin(2 * np.pi * 2 * timesteps)
    frequencies = get_self_correlating_frequencies(timesteps, signal)
    assert np.isclose(frequencies[0], 2.0, rtol=1e-2)

    # Test a signal with multiple self-correlating frequencies
    timesteps = np.linspace(0, 20, 2000)
    signal = np.sin(2 * np.pi * timesteps) + np.sin(2 * np.pi * 2 * timesteps)
    frequencies = get_self_correlating_frequencies(timesteps, signal)
    assert np.isclose(frequencies[0], 2.0, rtol=1e-2)
    assert np.isclose(frequencies[1], 1.0, rtol=1e-2)
