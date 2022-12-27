import numpy as np

from ntrfc.timeseries.classify_stationarity import classify_stationarity, min_max_normalize


def test_classify_stationarity():
    def generate_signal(amplitude, noiseamplitude, frequency, mean,  time):
        resolution = 20480
        step = (resolution * frequency ** -1) ** -1

        times = np.arange(0, time, step)
        noise = np.random.normal(0, noiseamplitude, len(times))

        values = amplitude * np.sin(frequency * (2 * np.pi) * times) + mean + noise
        return times, values

    deterministic_amplitude = 1
    noise_level = 0.01
    frequency = 10

    mean = 10
    time = 10
    # Calculate the time span in milliseconds

    timesteps,values = generate_signal(deterministic_amplitude,noise_level, frequency, mean, time)

    constant_level, deterministic_level, noise_level = classify_stationarity(timesteps,values)

    assert deterministic_level == True, "signal is supposed to be classified as deterministic"

    deterministic_amplitude = 0
    noise_level = 2
    frequency = 10

    mean = 0
    time = 10
    # Calculate the time span in milliseconds

    timesteps,values = generate_signal(deterministic_amplitude,noise_level, frequency, mean, time)

    constant_level, deterministic_level, noise_level = classify_stationarity(timesteps,values)

    assert noise_level == True, "signal is supposed to be classified as noisy"

    deterministic_amplitude = 0
    noise_level = 0.01
    frequency = 10

    mean = 10
    time = 10
    # Calculate the time span in milliseconds

    timesteps,values = generate_signal(deterministic_amplitude,noise_level, frequency, mean, time)

    constant_level, deterministic_level, noise_level = classify_stationarity(timesteps,values)

    assert noise_level == True, "signal is supposed to be classified as noisy"


    deterministic_amplitude = 0.2
    noise_level = 0.05
    frequency = 10

    mean = 0
    time = 10
    # Calculate the time span in milliseconds

    timesteps,values = generate_signal(deterministic_amplitude,noise_level, frequency, mean, time)

    constant_level, deterministic_level, noise_level = classify_stationarity(timesteps,values)

    assert deterministic_level == True, "signal is supposed to be classified as const"


def test_min_max_normalize():
    # Test with a simple signal
    signal = np.array([1, 2, 3])
    expected = np.array([0., 0.5, 1.])
    result = min_max_normalize(signal)
    assert np.allclose(result, expected), f'Error: {result}'

    # Test with a signal that is already normalized
    signal = np.array([0., 0.5, 1.])
    expected = np.array([0., 0.5, 1.])
    result = min_max_normalize(signal)
    assert np.allclose(result, expected), f'Error: {result}'

    # Test with a signal that contains negative values
    signal = np.array([-1, 0, 1])
    expected = np.array([0., 0.5, 1.])
    result = min_max_normalize(signal)
    assert np.allclose(result, expected), f'Error: {result}'

    # Test with a signal that contains extreme values
    signal = np.array([1, 2, 3, 1000])
    expected = np.array([0., 0.001001, 0.002002, 1.])
    result = min_max_normalize(signal)
    assert np.allclose(result, expected), f'Error: {result}'

    # Test with a signal that is all 0s
    signal = np.array([0, 0, 0])
    expected = np.array([0., 0., 0.])
    result = min_max_normalize(signal)
    assert np.allclose(result, expected), f'Error: {result}'
