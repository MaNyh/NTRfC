import os

import numpy as np
import pyvista as pv

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


def test_optimal_window_size():
    from ntrfc.timeseries.stationarity import optimal_window_size

    resolution = 10000
    num_periods = 8
    x = np.linspace(0, 2 * np.pi * num_periods, resolution)

    # Test with sine wave
    sine_wave = np.sin(x)
    _, window_size = optimal_window_size(sine_wave)
    assert window_size == resolution // num_periods * 2

    # Test with hyperbolic tangent
    tanh_wave = np.tanh(x * 2)
    _, window_size = optimal_window_size(tanh_wave)
    assert window_size == resolution * 0.05

    # Test with exponential decay
    exp_decay = np.e ** (-x * 60)
    _, window_size = optimal_window_size(exp_decay)
    assert window_size == resolution * 0.05


def test_stationarity_of_sine_wave():
    from ntrfc.timeseries.stationarity import estimate_stationarity
    from itertools import product

    amplitudes = [1]
    frequencies = [1]
    durations = [4, 20]
    means = [-1]

    test_cases = list(product(amplitudes, frequencies, durations, means))

    for amplitude, frequency, duration, mean in test_cases:
        timesteps, values = signal_generator(amplitude=amplitude, frequency=frequency, mean=mean, time=duration)
        stationary_index, _, _ = estimate_stationarity(values)

        assert timesteps[stationary_index] == 0, "Stationarity computation failed"


def test_step_function_stationarity():
    """
    This test proves that a pure step function can't be considered stationary.
    Because we normalize the signal, it does not matter how big the step is.
    This will change when we use noise in the signal, as we then have a real signal-to-noise ratio.
    """
    from ntrfc.timeseries.stationarity import estimate_stationarity
    import numpy as np

    resolution = 20000
    signal_amplitude = 1

    # Test with high signal-to-noise ratio (SNR)
    high_snr = 1e2
    step_size_high_snr = high_snr / signal_amplitude
    values_high_snr = signal_amplitude * np.ones(resolution)
    values_high_snr[resolution // 2:] = 1 + step_size_high_snr
    stationary_timestep_high_snr, _, _ = estimate_stationarity(values_high_snr)
    assert stationary_timestep_high_snr == resolution // 2

    # Test with low signal-to-noise ratio (SNR)
    low_snr = 1e-12
    step_size_low_snr = low_snr / signal_amplitude
    values_low_snr = signal_amplitude * np.ones(resolution)
    values_low_snr[resolution // 2:] = 1 + step_size_low_snr
    stationary_timestep_low_snr, _, _ = estimate_stationarity(values_low_snr)
    assert stationary_timestep_low_snr == resolution // 2


def test_transient_function_stationarity():
    """
    This test proves that a pure transient function can't be considered stationary.
    """
    from ntrfc.timeseries.stationarity import estimate_stationarity
    import numpy as np

    resolution = 20000

    high_values = np.linspace(0, 10, resolution)
    stationary_timestep, _, _ = estimate_stationarity(high_values)
    assert stationary_timestep is None

    low_values = np.linspace(0, 1e-6, resolution)
    stationary_timestep, _, _ = estimate_stationarity(low_values)
    assert stationary_timestep is None


def test_stationarity_with_ramp():
    from ntrfc.timeseries.stationarity import estimate_stationarity
    import numpy as np

    resolution = 20000
    noise_amplitude = 1e-8

    # High SNR test
    high_snr = 1e16
    high_ramp_amplitude = noise_amplitude * high_snr
    noise = np.random.normal(-1, 1, resolution) * noise_amplitude
    values_high_snr = np.zeros(resolution) + noise
    values_high_snr[:resolution // 2] += np.linspace(high_ramp_amplitude, 0, resolution // 2)
    stationary_timestep_high_snr, _, _ = estimate_stationarity(values_high_snr)
    assert stationary_timestep_high_snr in range(9000, 10000)

    # Low SNR test
    low_snr = 1e-13
    low_ramp_amplitude = noise_amplitude * low_snr
    values_low_snr = np.zeros(resolution) + noise
    values_low_snr[:resolution // 2] += np.linspace(low_ramp_amplitude, 0, resolution // 2)
    stationary_timestep_low_snr, _, _ = estimate_stationarity(values_low_snr)
    assert stationary_timestep_low_snr == 0


def test_stationarity_with_noise():
    from ntrfc.timeseries.stationarity import estimate_stationarity
    import numpy as np

    small_sample_size = 100
    large_sample_size = 40000

    # Test with small sample size
    small_sample_values = np.random.normal(-1, 1, small_sample_size)
    stationary_timestep, _, _ = estimate_stationarity(small_sample_values)
    assert stationary_timestep is None

    # Test with large sample size
    large_sample_values = np.random.normal(-1, 1, large_sample_size)
    stationary_timestep, _, _ = estimate_stationarity(large_sample_values)
    assert stationary_timestep == 0


def test_stationarity_with_abating_sine_wave():
    from ntrfc.timeseries.stationarity import estimate_stationarity
    from ntrfc.math.methods import reldiff
    from itertools import product

    amplitudes = [0.3]
    frequencies = [6]
    durations = [20]
    means = [-1]
    abate_factors = [2, 5]

    test_cases = list(product(amplitudes, frequencies, durations, means, abate_factors))

    for amplitude, frequency, duration, mean, abate in test_cases:
        timesteps, values = signal_generator(amplitude=amplitude, frequency=frequency, mean=mean, time=duration,
                                             abate=abate)
        stationary_timestep, _, _ = estimate_stationarity(values)

        expected_stationarity_limit = -np.log(0.001) / abate
        expected_stationary_time = timesteps[-1] - expected_stationarity_limit
        computed_stationary_time = timesteps[-1] - timesteps[stationary_timestep]

        # plt.figure()
        # plt.plot(timesteps, values)
        # plt.axvline(timesteps[stationary_timestep], color="green")
        # plt.axvline(expected_stationarity_limit, color="red", label="computed")
        # plt.legend()
        # plt.show()
        assert 0.05 > reldiff(computed_stationary_time, expected_stationary_time), "computation failed"


def test_stationarity_with_abating_sine_noise():
    from ntrfc.timeseries.stationarity import estimate_stationarity
    from ntrfc.math.methods import reldiff
    from itertools import product

    amplitudes = [0.1]
    noise_amplitudes = [0.01]
    frequencies = [6]
    durations = [40]
    means = [-1]
    abate_factors = [3]

    test_cases = list(product(amplitudes, noise_amplitudes, frequencies, durations, means, abate_factors))

    for amplitude, noise_amplitude, frequency, duration, mean, abate in test_cases:
        timesteps, values = signal_generator(
            amplitude=amplitude, noise_amplitude=noise_amplitude,
            frequency=frequency, mean=mean, time=duration, abate=abate
        )
        stationary_timestep, _, _ = estimate_stationarity(values)

        expected_stationarity_limit = -np.log(0.001) / abate
        expected_stationary_time = timesteps[-1] - expected_stationarity_limit
        computed_stationary_time = timesteps[-1] - timesteps[stationary_timestep]

        assert 0.05 >= reldiff(computed_stationary_time, expected_stationary_time), "computation failed"


def test_stationarity_with_transient_signal():
    from ntrfc.timeseries.stationarity import estimate_stationarity
    from itertools import product

    def generate_abating_sine_wave(amplitude, frequency, mean, duration):
        resolution = 36
        step = (1 / frequency) / resolution
        times = np.arange(0, duration, step)
        values = (amplitude + frequency * (2 * np.pi) * times) * np.sin(frequency * (2 * np.pi) * times) + mean
        return times, values

    amplitudes = [0.1]
    frequencies = [2]
    durations = [40]
    means = [-2]

    test_cases = list(product(amplitudes, frequencies, durations, means))
    for amplitude, frequency, duration, mean in test_cases:
        times, values = generate_abating_sine_wave(amplitude=amplitude, frequency=frequency, mean=mean,
                                                   duration=duration)
        stationary_index, _, _ = estimate_stationarity(values)

        assert stationary_index is None


def test_stationarity_with_nonlinear_signal():
    from ntrfc.timeseries.stationarity import estimate_stationarity
    import numpy as np

    # not using a seed for numpy will result in non-reproducible results
    # therefore, when noise is used a big range of return is allowed
    resolution = 20000
    base_values = np.zeros(resolution)
    noise = np.random.normal(-1, 1, resolution) * 1e-6

    # Test with no noise
    base_values[:resolution // 2] += np.exp(-np.linspace(0, 10, resolution // 2))
    stationary_timestep, _, _ = estimate_stationarity(base_values)
    assert stationary_timestep == resolution // 2

    # Test with low noise
    low_noise_values = np.zeros(resolution) + noise
    low_noise_values[:resolution // 2] += np.exp(-np.linspace(0, 10, resolution // 2))
    stationary_timestep_low_noise, _, _ = estimate_stationarity(low_noise_values)
    assert stationary_timestep_low_noise in range(5000, 10001)

    # Test with high noise
    high_noise_values = np.zeros(resolution) + noise * 3
    high_noise_values[:resolution // 2] += np.exp(-np.linspace(0, 10, resolution // 2))
    stationary_timestep_high_noise, _, _ = estimate_stationarity(high_noise_values)
    assert stationary_timestep_high_noise in range(stationary_timestep_low_noise - 200,
                                                   stationary_timestep_low_noise + 200)


def test_stationarity_with_constant_signal():
    from ntrfc.timeseries.stationarity import estimate_stationarity
    import numpy as np

    resolution = 20000
    constant_signal = np.ones(resolution)
    stationary_timestep, _, _ = estimate_stationarity(constant_signal)
    assert stationary_timestep == 0


def test_stationarity_with_abating_signals(tmp_path):
    from ntrfc.timeseries.stationarity import estimate_stationarity
    from itertools import product
    from ntrfc.math.methods import reldiff

    noise_amplitudes = [0.01]
    durations = [60]
    means = [-1]
    abate_factors = [3, 2]

    test_cases = list(product(noise_amplitudes, durations, means, abate_factors))

    for noise_amplitude, duration, mean, abate in test_cases:
        timesteps, values = signal_generator(noise_amplitude=noise_amplitude, mean=mean, abate=abate, time=duration)
        stationary_timestep, _, _ = estimate_stationarity(values, create_plot_dir=tmp_path)

        expected_stationarity_limit = -np.log(0.001) / abate
        expected_stationary_time = timesteps[-1] - expected_stationarity_limit
        computed_stationary_time = timesteps[-1] - timesteps[stationary_timestep]

        assert 0.05 >= reldiff(computed_stationary_time, expected_stationary_time), "computation failed"


def test_last_coherent_interval_function():
    all_true_array = np.ones(100)
    assert last_coherent_interval(all_true_array, 10, 0.95) == 100

    two_sigma_true_array = np.ones(100)
    for _ in range(5):
        random_index = np.random.randint(0, 100)
        two_sigma_true_array[random_index] = 0

    assert last_coherent_interval(two_sigma_true_array, 10, 0.95) == 100

    half_true_array = np.ones(100)
    half_true_array[50:] = False
    assert last_coherent_interval(half_true_array, 10, 1) == 50


def test_stationarity_with_noise_convergence():
    from ntrfc.timeseries.stationarity import estimate_stationarity

    num_repetitions = 24
    sample_size = 5000

    for seed in range(num_repetitions):
        np.random.seed(seed)
        noise = np.random.normal(-0.1, 0.1, sample_size)
        stationary_index, _, _ = estimate_stationarity(noise)
        assert stationary_index is not None, f"Computation failed, should not return None for sample size > {sample_size} in run {seed}"
