import numpy as np
import matplotlib.pyplot as plt
import itertools

from ntrfc.timeseries.stationarity import estimate_stationarity


def mean_squared_error(x, y):
    return np.mean((x - y) ** 2)


def test_stationarity():
    def sine(x, f, a):
        return np.sin(x * f) * a

    def tanh(x, c, a):
        return np.tanh(x * c) * a

    def step(x, c, a):
        return np.piecewise(x, [x < c, x >= 2 / 2 * np.pi], [0, 1]) * a

    def linear(x, g):
        return x * g

    def dirac(x, c, a):
        return a * np.isclose(x,c)

    def euler(x, c, a):
        return np.e ** (-x * c) * a

    def noise(x, a):
        res = np.random.rand(len(x)) * a
        np.random.shuffle(res)
        return res

    sine_frequencies = [24, 100]
    sine_amplitudes = np.linspace(0.5, 4, 1)
    sine_resolutions = [8000, 13000]

    res = []
    for frequency, amplitude, resolution in itertools.product(sine_frequencies, sine_amplitudes, sine_resolutions):
        # Do something with frequency and amplitude
        x = x_vals(resolution)
        f = sine(x, frequency, amplitude)
        ans = estimate_stationarity(f)

        res.append(ans / len(x))
        plt.plot(f)
        plt.vlines(ans, ymin=min(f), ymax=max(f), colors=["red"])
        plt.grid()
        plt.show()

    plt.figure()
    plt.hist(res)
    plt.show()

    tanh_c = [3]
    tanh_amplitudes = np.linspace(0.2, 1, 1)
    tanh_resolutions = [1000, 10000, 20000]

    res = []
    for frequency, amplitude, resolution in itertools.product(tanh_c, tanh_amplitudes, tanh_resolutions):
        # Do something with frequency and amplitude
        x = x_vals(resolution)
        f = tanh(x, frequency, amplitude)
        ans = estimate_stationarity(f)

        res.append(ans / len(x))
        plt.plot(f)
        plt.vlines(ans, ymin=min(f), ymax=max(f), colors=["red"])
        plt.grid()
        plt.show()

    plt.figure()
    plt.hist(res)
    plt.show()

    euler_c = [10]
    euler_amplitudes = np.linspace(0.2, 1, 1)
    euler_resolutions = [10000,20000]

    res = []
    for frequency, amplitude, resolution in itertools.product(euler_c, euler_amplitudes, euler_resolutions):
        # Do something with frequency and amplitude
        x = x_vals(resolution)
        f = euler(x, frequency, amplitude)
        ans = estimate_stationarity(f)

        res.append(ans / len(x))
        plt.plot(f)
        plt.vlines(ans, ymin=min(f), ymax=max(f), colors=["red"])
        plt.grid()
        plt.show()

    plt.figure()
    plt.hist(res)
    plt.show()

    step_c = [np.pi / 6]
    step_amplitudes = np.linspace(0.2, 1, 1)

    step_resolutions = [1000, 10000]
    res = []
    for frequency, amplitude, resolution in itertools.product(step_c, step_amplitudes, step_resolutions):
        # Do something with frequency and amplitude
        x = x_vals(resolution)
        f = step(x, frequency, amplitude)
        ans = estimate_stationarity(f)

        res.append(ans / len(x))
        plt.plot(f)
        plt.vlines(ans, ymin=min(f), ymax=max(f), colors=["red"])
        plt.grid()
        plt.show()

    plt.figure()
    plt.hist(res)
    plt.show()

    noise_amplitudes = np.linspace(0.2, 1, 2)

    noise_resolutions = [6000, 20000]
    res = []
    for amplitude, resolution in itertools.product(noise_amplitudes, noise_resolutions):
        # Do something with frequency and amplitude
        x = x_vals(resolution)
        f = noise(x, amplitude)
        ans = estimate_stationarity(f)

        res.append(ans / len(x))
        plt.plot(f)
        plt.vlines(ans, ymin=min(f), ymax=max(f), colors=["red"])
        plt.grid()
        plt.show()

    plt.figure()
    plt.hist(res)
    plt.show()

    df = [1]
    res = []
    for g in df:
        resolution = 10000
        # Do something with frequency and amplitude
        x = x_vals(resolution)
        f = linear(x, g)
        ans = estimate_stationarity(f)
        if ans != False:
            res.append(ans / len(x))
        else:
            res.append(1.1)
        plt.plot(f)
        plt.vlines(ans, ymin=min(f), ymax=max(f), colors=["red"])
        plt.grid()
        plt.show()

    plt.figure()
    plt.hist(res)
    plt.show()

    df = [1]
    res = []
    for g in df:
        resolution = 10001
        # Do something with frequency and amplitude
        x = x_vals(resolution)/(2*np.pi)
        f = dirac(x, 0.5,0.5)
        ans = estimate_stationarity(f)
        if ans != False:
            res.append(ans / len(x))
        else:
            res.append(1.1)
        plt.plot(f)
        plt.vlines(ans, ymin=min(f), ymax=max(f), colors=["red"])
        plt.grid()
        plt.show()

    plt.figure()
    plt.hist(res)
    plt.show()


def x_vals(res):
    resolution = res
    start = 0
    end = 2 * np.pi
    x = np.linspace(start, end, resolution)
    return x


