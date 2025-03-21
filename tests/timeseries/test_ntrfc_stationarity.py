import os

import pyvista as pv
import numpy as np

from ntrfc.timeseries.stationarity import last_coherent_interval

ON_CI = 'CI' in os.environ

if ON_CI:
    pv.start_xvfb()


def signal_generator(amplitude=1, frequency=1, mean=0, abate=None, noise_amplitude=0.0, time=10.0, resolution=128):
    step = time * (frequency * resolution)
    times = np.linspace(0, time, step)
    noise = np.random.normal(-1, 1, len(times)) * noise_amplitude if noise_amplitude else 0
    decay = np.e ** -(times * abate) if abate else 0
    values = amplitude * np.sin(frequency * times * (np.pi * 2)) + mean + decay + noise
    return times, values


def test_optimal_timewindow():
    from ntrfc.timeseries.stationarity import optimal_window_size
    # sine
    res = 10000

    nper = 8
    x = np.linspace(0, 2 * np.pi * nper, res)

    # sin
    # we have four periods, at least one period should be captured
    # thats res // 4 as a return
    ysin = np.sin(x)
    opt_window, opt_window_size, _ = optimal_window_size(ysin)

    assert opt_window_size == res // nper * 2
    # tanh
    tanh = np.tanh(x * 2)
    opt_window, opt_window_size, _ = optimal_window_size(tanh)
    assert opt_window_size == res * 0.05
    # euler
    eul = np.e ** (-x * 60)
    opt_window, opt_window_size, _ = optimal_window_size(eul)

    assert opt_window_size == res * 0.05


def test_stationarity_stationarysine():
    # T4

    from ntrfc.timeseries.stationarity import estimate_stationarity
    from itertools import product

    test_amplitudes = [1]  # 1,10
    test_frequencies = [1]  # 1,10
    test_times = [4, 20]  # 10, 60
    test_mean = [-1]  # -1,0,1

    test_configs = list(product(test_amplitudes, test_frequencies, test_times, test_mean))

    for amplitude, frequency, time, mean in test_configs:
        timesteps, values = signal_generator(amplitude=amplitude, frequency=frequency, mean=mean, time=time)
        stationary_timestep, _, _ = estimate_stationarity(values)

        assert 0 == timesteps[stationary_timestep], "computation failed"


def test_stationarity_step():
    """

    this test proves that a pure step function cant be considered stationary
    because we normalize the signal, it does not matter how big the step is
    this will change when we use noise in the signal, as we then have a real signal-to-noise ratio
    """
    from ntrfc.timeseries.stationarity import estimate_stationarity
    import numpy as np

    res = 20000

    snr_false = 1e2
    signal_amplitude = 1
    values_stationary = signal_amplitude * np.ones(res)
    stepsize = snr_false / signal_amplitude
    values_stationary[res // 2:] = 1 + stepsize
    stationary_timestep_false, _, _ = estimate_stationarity(values_stationary)

    assert stationary_timestep_false == 10000

    snr = 1e-12
    stepsize = snr / signal_amplitude
    values_stationary = np.ones(res) * signal_amplitude
    values_stationary[res // 2:] = 1 + stepsize
    stationary_timestep_true, _, _ = estimate_stationarity(values_stationary)
    assert stationary_timestep_true == 10000


def test_stationarity_transientonly_linear():
    """
    this test proves that a pure transient function cant be considered stationary
    T10
    """
    from ntrfc.timeseries.stationarity import estimate_stationarity
    import numpy as np

    res = 20000

    values_high = np.linspace(0, 10, res)
    stationary_timestep, _, _ = estimate_stationarity(values_high)
    assert stationary_timestep is None

    values_low = np.linspace(0, 1e-6, res)
    stationary_timestep, _, _ = estimate_stationarity(values_low)
    assert stationary_timestep is None


def test_stationarity_ramp():
    from ntrfc.timeseries.stationarity import estimate_stationarity
    import numpy as np

    res = 20000
    noise_amplitude = 1e-8
    snr = 1e16
    ramp_amplitude = noise_amplitude * snr

    noise = np.random.normal(-1, 1, res) * noise_amplitude

    values_stationary = np.zeros(res) + noise
    values_stationary[:res // 2] = values_stationary[:res // 2] + np.linspace(ramp_amplitude, 0, res // 2)
    stationary_timestep_true, _, _ = estimate_stationarity(values_stationary)
    assert stationary_timestep_true in range(9000, 10000)

    snr = 1e-13
    ramp_amplitude = noise_amplitude * snr
    values_stationary_new = np.zeros(res) + noise

    values_stationary_new[:res // 2] = values_stationary_new[:res // 2] + np.linspace(ramp_amplitude, 0, res // 2)
    stationary_timestep_false, _, _ = estimate_stationarity(values_stationary_new)

    assert stationary_timestep_false == 0


def test_stationarity_noise():
    from ntrfc.timeseries.stationarity import estimate_stationarity
    import numpy as np

    res = 100

    values = np.random.normal(-1, 1, res)
    stationary_timestep, _, _ = estimate_stationarity(values)
    assert stationary_timestep is None

    res = 40000

    values = np.random.normal(-1, 1, res)
    stationary_timestep, _, _ = estimate_stationarity(values)
    assert stationary_timestep == 0


def test_stationarity_uncertainties_abatingsine():
    from ntrfc.timeseries.stationarity import estimate_stationarity
    from ntrfc.math.methods import reldiff
    from itertools import product

    test_amplitudes = [0.3]
    test_frequencies = [6]
    test_times = [20]
    test_mean = [-1]
    test_abate = [2, 5]

    test_configs = list(product(test_amplitudes, test_frequencies, test_times, test_mean, test_abate))

    for amplitude, frequency, time, mean, abate in test_configs:
        timesteps, values = signal_generator(amplitude=amplitude, frequency=frequency, mean=mean, time=time,
                                             abate=abate)
        stationary_timestep, _, _ = estimate_stationarity(values)

        well_computed_stationarity_limit = -np.log(0.001) / abate
        well_computed_stationary_time = timesteps[-1] - well_computed_stationarity_limit
        stationary_time = timesteps[-1] - timesteps[stationary_timestep]

        # plt.figure()
        # plt.plot(timesteps, values)
        # plt.axvline(timesteps[stationary_timestep], color="green")
        # plt.axvline(well_computed_stationarity_limit, color="red", label="computed")
        # plt.legend()
        # plt.show()
        assert 0.05 > reldiff(stationary_time, well_computed_stationary_time), "computation failed"


def test_stationarity_uncertainties_abatingsinenoise(verbose=False):
    from ntrfc.timeseries.stationarity import estimate_stationarity
    from ntrfc.math.methods import reldiff
    from itertools import product

    test_amplitudes = [0.1]
    test_noiseamplitude = [0.01]
    test_frequencies = [6]
    test_times = [40]
    test_mean = [-1]
    test_abate = [3]

    test_configs = list(
        product(test_amplitudes, test_noiseamplitude, test_frequencies, test_times, test_mean, test_abate))

    for amplitude, noiseamplitude, frequency, time, mean, abate in test_configs:
        timesteps, values = signal_generator(amplitude=amplitude, noise_amplitude=noiseamplitude,
                                             frequency=frequency, mean=mean, time=time,
                                             abate=abate)
        stationary_timestep, _, _ = estimate_stationarity(values)

        well_computed_stationarity_limit = -np.log(0.001) / abate
        well_computed_stationary_time = timesteps[-1] - well_computed_stationarity_limit
        stationary_time = timesteps[-1] - timesteps[stationary_timestep]
        # if verbose:
        #     plt.figure()
        #     plt.plot(timesteps, values)
        #     plt.axvline(timesteps[stationary_timestep], color="green")
        #     plt.axvline(well_computed_stationarity_limit, color="red")
        #     plt.show()
        assert 0.05 >= reldiff(stationary_time, well_computed_stationary_time), "computation failed"


def test_stationarity_transientonly(verbose=False):
    from ntrfc.timeseries.stationarity import estimate_stationarity
    from itertools import product
    import matplotlib.pyplot as plt

    def signalgen_abatingsine(amplitude, frequency, mean, time):
        resolution = 36
        step = (1 / frequency) / resolution

        times = np.arange(0, time, step)

        values = (amplitude + frequency * (2 * np.pi) * times) * np.sin(frequency * (2 * np.pi) * times) + mean
        return times, values

    test_amplitudes = [0.1]
    test_frequencies = [2]
    test_times = [40]
    test_mean = [-2]

    test_configs = list(
        product(test_amplitudes, test_frequencies, test_times, test_mean))
    for amplitude, frequency, time, mean in test_configs:

        timesteps, values = signalgen_abatingsine(amplitude=amplitude,
                                                  frequency=frequency, mean=mean, time=time)

        statidx, _, _ = estimate_stationarity(values)

        if verbose:
            plt.figure()
            plt.plot(timesteps, values)
            plt.show()
        assert statidx is None


def test_stationarity_nonlinear():
    from ntrfc.timeseries.stationarity import estimate_stationarity
    import numpy as np

    # not using a seed for numpy will result not reproducible results
    # therefore, when noise is used a big range of return is allowed
    res = 20000
    values = np.zeros(res)
    noise = np.random.normal(-1, 1, res) * 1e-6
    values[:res // 2] = values[:res // 2] + np.e ** -np.linspace(0, 10, res // 2)
    stationary_timestep, _, _ = estimate_stationarity(values)
    assert stationary_timestep == res // 2

    values_new = np.zeros(res) + noise
    values_new[:res // 2] = values_new[:res // 2] + np.e ** -np.linspace(0, 10, res // 2)
    stationary_timestep_new, _, _ = estimate_stationarity(values_new)
    assert stationary_timestep_new in range(5000, 10001)

    values_highnoise = np.zeros(res) + noise * 3
    values_highnoise[:res // 2] = values_highnoise[:res // 2] + np.e ** -np.linspace(0, 10, res // 2)
    stationary_timestep_new_new, _, _ = estimate_stationarity(values_highnoise)
    assert stationary_timestep_new_new in range(stationary_timestep_new - 200, stationary_timestep_new + 200)


def test_stationarity_constant():
    # T1
    from ntrfc.timeseries.stationarity import estimate_stationarity
    import numpy as np

    res = 20000
    values_stationary = np.ones(res)
    stationary_timestep, _, _ = estimate_stationarity(values_stationary)
    assert stationary_timestep == 0


def test_stationarity_uncertainties_abating(tmp_path, verbose=False):
    from ntrfc.timeseries.stationarity import estimate_stationarity
    from itertools import product
    import matplotlib.pyplot as plt
    from ntrfc.math.methods import reldiff

    test_noiseamplitude = [0.01]
    test_times = [60]
    test_mean = [-1]
    test_abate = [3, 2]

    test_configs = list(product(test_noiseamplitude, test_times, test_mean, test_abate))

    for noiseamplitude, time, mean, abate in test_configs:

        timesteps, values = signal_generator(noise_amplitude=noiseamplitude, mean=mean, abate=abate, time=time)
        stationary_timestep, _, _ = estimate_stationarity(values, create_plot_dir=tmp_path)

        well_computed_stationarity_limit = -np.log(0.001) / abate
        well_computed_stationary_time = timesteps[-1] - well_computed_stationarity_limit
        stationary_time = timesteps[-1] - timesteps[stationary_timestep]
        if verbose:
            plt.figure()
            plt.plot(timesteps, values)
            plt.axvline(timesteps[stationary_timestep], color="green")
            plt.axvline(well_computed_stationarity_limit, color="red")
            plt.show()
        assert 0.05 >= reldiff(stationary_time, well_computed_stationary_time), "computation failed"


def test_last_coherent_interval():
    all_true = np.ones(100)
    assert last_coherent_interval(all_true, 10, 0.95) == 100

    twosigma_true = np.ones(100)
    for i in range(5):
        idx = np.random.randint(0, 100)
        twosigma_true[idx] = 0

    assert last_coherent_interval(twosigma_true, 10, 0.95) == 100

    half_true = np.ones(100)
    half_true[50:] = False
    assert last_coherent_interval(half_true, 10, 1) == 50


def test_statistical_convergence():
    from ntrfc.timeseries.stationarity import estimate_stationarity

    repetitions = 4
    for i in range(repetitions):
        np.random.seed(i)

        res = 30000
        noise = np.random.normal(-0.1, 0.1, res)
        idx, _, _ = estimate_stationarity(noise)
        assert idx != None, f"computation failed, should not return None for samplesize > {res} in run {i}"
