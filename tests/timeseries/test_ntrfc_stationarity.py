import os

import pyvista as pv

ON_CI = 'CI' in os.environ

if ON_CI:
    pv.start_xvfb()


def test_optimal_timewindow(verbose=False):
    from ntrfc.timeseries.stationarity import optimal_window_size
    import numpy as np
    # sine
    res = 10000

    nper = 8
    x = np.linspace(0, 2 * np.pi * nper, res)
    min_interval = 0.05
    max_interval = 0.25
    # sin
    # we have four periods, at least one period should be captured
    # thats res // 4 as a return
    ysin = np.sin(x)
    opt_window, opt_window_size, nperiods = optimal_window_size(ysin, min_interval, max_interval)
    assert opt_window_size == res // nper * nperiods
    assert nperiods == 2
    # tanh
    tanh = np.tanh(x * 2)
    opt_window, opt_window_size, nperiods = optimal_window_size(tanh, min_interval, max_interval)
    assert opt_window_size / 2 == res * min_interval
    assert nperiods == 0
    # euler
    eul = np.e ** (-x * 60)
    opt_window, opt_window_size, nperiods = optimal_window_size(eul, min_interval, max_interval)
    assert opt_window_size / 2== res * min_interval
    assert nperiods == 0


def test_stationarity_uncertainties_stationarysine(verbose=False):
    from ntrfc.timeseries.stationarity import estimate_stationarity
    from ntrfc.math.methods import reldiff
    import numpy as np
    from itertools import product
    import matplotlib.pyplot as plt

    def signalgen_sine(amplitude, frequency, mean, time):
        frequency_resolution = 256
        tau = frequency ** -1

        step = tau / frequency_resolution
        times = np.arange(0, time, step)
        values = amplitude * np.sin(frequency * (2 * np.pi) * times) + mean
        return times, values

    test_amplitudes = [1]  # 1,10
    test_frequencies = [1]  # 1,10
    test_times = [20]  # 10, 60
    test_mean = [-1]  # -1,0,1

    maxperiods = -1
    minperiods = np.inf

    test_configs = list(product(test_amplitudes, test_frequencies, test_times, test_mean))

    for amplitude, frequency, time, mean in test_configs:
        equals_periods = time / frequency ** -1
        if equals_periods > maxperiods:
            maxperiods = int(equals_periods)
        if minperiods > equals_periods:
            minperiods = int(equals_periods)
        timesteps, values = signalgen_sine(amplitude=amplitude, frequency=frequency, mean=mean, time=time)
        stationary_timestep = estimate_stationarity(values)
        analytic_stationary_limit = 0.0
        if verbose:
            plt.figure()
            plt.plot(timesteps, values)
            plt.axvline(stationary_timestep)
            plt.show()
        assert 0.05 > reldiff(time - analytic_stationary_limit,
                              time - timesteps[stationary_timestep]), "computation failed"


def test_stationarity_uncertainties_abatingsine(verbose=False):
    from ntrfc.timeseries.stationarity import estimate_stationarity
    from ntrfc.math.methods import reldiff
    import numpy as np
    from itertools import product
    import matplotlib.pyplot as plt

    def signalgen_abatingsine(amplitude, frequency, mean, abate, time):
        resolution = 64
        step = (1 / frequency) / resolution
        times = np.arange(0, time, step)
        values = amplitude * np.sin(frequency * (2 * np.pi) * times) + mean + np.e ** -(times * abate)
        return times, values

    test_amplitudes = [0.3]
    test_frequencies = [6]
    test_times = [20]
    test_mean = [-1]
    test_abate = [2, 5]

    test_configs = list(product(test_amplitudes, test_frequencies, test_times, test_mean, test_abate))

    for amplitude, frequency, time, mean, abate in test_configs:

        timesteps, values = signalgen_abatingsine(amplitude=amplitude, frequency=frequency, mean=mean, time=time,
                                                  abate=abate)
        stationary_timestep = estimate_stationarity(values)

        well_computed_stationarity_limit = -np.log(0.01) / abate
        well_computed_stationary_time = timesteps[-1] - well_computed_stationarity_limit
        stationary_time = timesteps[-1] - timesteps[stationary_timestep]
        if verbose:
            plt.figure()
            plt.plot(timesteps, values)
            plt.axvline(timesteps[stationary_timestep], color="green")
            plt.axvline(well_computed_stationarity_limit, color="red", label="computed")
            plt.legend()
            plt.show()
        assert 0.05 > reldiff(stationary_time, well_computed_stationary_time), "computation failed"


def test_stationarity_uncertainties_abatingsinenoise(verbose=False):
    from ntrfc.timeseries.stationarity import estimate_stationarity
    from ntrfc.math.methods import reldiff
    import numpy as np
    from itertools import product
    import matplotlib.pyplot as plt

    def signalgen_abatingsine(amplitude, noiseamplitude, frequency, mean, abate, time):
        resolution = 64
        step = (1 / frequency) / resolution

        times = np.arange(0, time, step)
        noise = np.random.normal(-1, 1, len(times)) * noiseamplitude

        values = amplitude * np.sin(frequency * (2 * np.pi) * times) + mean + np.e ** -(times * abate) + noise
        return times, values

    test_amplitudes = [0.1]
    test_noiseamplitude = [0.01]
    test_frequencies = [6]
    test_times = [40]
    test_mean = [-1]
    test_abate = [3]

    test_configs = list(
        product(test_amplitudes, test_noiseamplitude, test_frequencies, test_times, test_mean, test_abate))

    for amplitude, noiseamplitude, frequency, time, mean, abate in test_configs:

        timesteps, values = signalgen_abatingsine(amplitude=amplitude, noiseamplitude=noiseamplitude,
                                                  frequency=frequency, mean=mean, time=time,
                                                  abate=abate)
        stationary_timestep = estimate_stationarity(values)

        well_computed_stationarity_limit = -np.log(0.01) / abate
        well_computed_stationary_time = timesteps[-1] - well_computed_stationarity_limit
        stationary_time = timesteps[-1] - timesteps[stationary_timestep]
        if verbose:
            plt.figure()
            plt.plot(timesteps, values)
            plt.axvline(timesteps[stationary_timestep], color="green")
            plt.axvline(well_computed_stationarity_limit, color="red")
            plt.show()
        assert 0.05 >= reldiff(stationary_time, well_computed_stationary_time), "computation failed"


def test_stationarity_transientonly(verbose=False):
    from ntrfc.timeseries.stationarity import estimate_stationarity
    import numpy as np
    from itertools import product
    import matplotlib.pyplot as plt

    def signalgen_abatingsine(amplitude, frequency, mean, time):
        resolution = 36
        step = (1 / frequency) / resolution

        times = np.arange(0, time, step)

        values = (amplitude + frequency * (2 * np.pi) * times / 1600) * np.sin(frequency * (2 * np.pi) * times) + mean
        return times, values

    test_amplitudes = [0.1]
    test_frequencies = [6]
    test_times = [40]
    test_mean = [-2]

    test_configs = list(
        product(test_amplitudes, test_frequencies, test_times, test_mean))
    for amplitude, frequency, time, mean in test_configs:

        timesteps, values = signalgen_abatingsine(amplitude=amplitude,
                                                  frequency=frequency, mean=mean, time=time)

        statidx = estimate_stationarity(values)

        if verbose:
            plt.figure()
            plt.plot(timesteps, values)
            plt.show()
        assert statidx == False


def test_stationarity_transientonly():
    from ntrfc.timeseries.stationarity import estimate_stationarity
    import numpy as np

    res = 20000

    values = np.linspace(0, 10, res)
    stationary_timestep = estimate_stationarity(values)
    assert stationary_timestep == False


def test_stationarity_uncertainties_abating(verbose=False):
    from ntrfc.timeseries.stationarity import estimate_stationarity
    import numpy as np
    from itertools import product
    import matplotlib.pyplot as plt

    from ntrfc.math.methods import reldiff
    def signalgen_abating(noiseamplitude, mean, abate, time):
        resolution = 20000
        step = (time / resolution)

        times = np.arange(0, time, step)
        noise = np.random.normal(-1, 1, len(times)) * noiseamplitude

        values = mean + np.e ** -(times * abate) + noise
        return times, values

    test_noiseamplitude = [0.01]
    test_times = [30]
    test_mean = [-1]
    test_abate = [3, 2]

    test_configs = list(product(test_noiseamplitude, test_times, test_mean, test_abate))

    for noiseamplitude, time, mean, abate in test_configs:

        timesteps, values = signalgen_abating(noiseamplitude=noiseamplitude, mean=mean, abate=abate, time=time)
        stationary_timestep = estimate_stationarity(values)

        well_computed_stationarity_limit = -np.log(0.01) / abate
        well_computed_stationary_time = timesteps[-1] - well_computed_stationarity_limit
        stationary_time = timesteps[-1] - timesteps[stationary_timestep]
        if verbose:
            plt.figure()
            plt.plot(timesteps, values)
            plt.axvline(timesteps[stationary_timestep], color="green")
            plt.axvline(well_computed_stationarity_limit, color="red")
            plt.show()
        assert 0.05 >= reldiff(stationary_time, well_computed_stationary_time), "computation failed"
