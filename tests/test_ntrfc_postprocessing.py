import numpy as np

from ntrfc.postprocessing.timeseries.stationary_signal_check import parsed_timeseries_analysis



def constant_signal(numtimesteps, time):
    timesteps = np.linspace(0, 1, numtimesteps) * time
    signal = np.ones(len(timesteps))
    return timesteps, signal


def ramp_signal(numtimesteps, time):
    timesteps = np.linspace(0, 1, numtimesteps) * time
    signal = np.linspace(0, 1, numtimesteps) * time
    return timesteps, signal


def ramptoconst_signal(numtimesteps, time):
    timesteps = np.linspace(0, 1, numtimesteps) * time
    signal = np.linspace(0, 2, numtimesteps) * time
    signal = np.clip(signal, a_min=0, a_max=1)
    return timesteps, signal


def sine_signal(numtimesteps, time):
    timesteps = np.linspace(0, 1, numtimesteps) * time
    omega = 1000
    signal = np.sin(timesteps * omega) + 1
    return timesteps, signal


def noise_signal(numtimesteps, time):
    mu, sigma = 0, 0.2  # mean and standard deviation
    s = np.random.normal(mu, sigma, size=numtimesteps)

    Dt = time / numtimesteps
    timesteps = np.linspace(0, time, numtimesteps)
    # dt is the timescale of the noisy signal --> emulated length scale!
    dt = time / 1000
    timescale = int(dt / Dt)
    weights = np.repeat(1.0, timescale) / timescale
    signal = np.convolve(s, weights, 'same') + 1
    return timesteps, signal


def convergentsine_signal(numtimesteps, time):
    omega = 1000
    timesteps = np.linspace(0, time, numtimesteps)
    signal = -3 * np.sin(timesteps*omega) / timesteps / 10
    signal += 1
    signal = np.nan_to_num(signal,0)
    return timesteps, signal


def tanh_signal(numtimesteps, time):
    timesteps = np.linspace(0, time, numtimesteps)
    stationary = time/10
    tanhsignal = np.tanh(timesteps * stationary ** -1)
    return timesteps, tanhsignal


def tanh_sin_signal(numtimesteps, time):
    timesteps = np.linspace(0, time, numtimesteps)
    stationary = time/4
    tanhsignal = np.tanh(timesteps * stationary ** -1)

    timesteps = np.linspace(0, 1, numtimesteps) * time
    omega = 10000 * np.random.rand()
    signal = np.sin(timesteps * omega) + 1

    return timesteps, signal*0.3+tanhsignal


def tanh_sin_noise_signal(numtimesteps, time):
    timesteps = np.linspace(0, time, numtimesteps)
    stationary = time/4
    tanhsignal = np.tanh(timesteps * stationary ** -1)

    timesteps = np.linspace(0, 1, numtimesteps) * time
    omega = 1000
    signal = np.sin(timesteps * omega) + 1

    mu, sigma = 0, 0.2  # mean and standard deviation
    s = np.random.normal(mu, sigma, size=numtimesteps)

    Dt = time / numtimesteps
    # dt is the timescale of the noisy signal --> emulated length scale!
    dt = time / 1000
    timescale = int(dt / Dt)
    weights = np.repeat(1.0, timescale) / timescale
    noise = np.convolve(s, weights, 'same') + 1

    omega = 1000
    timesteps = np.linspace(0, time, numtimesteps)
    nsignal = -3 * np.sin(timesteps*omega) / timesteps / 10
    nsignal += 1
    nsignal = np.nan_to_num(signal,0)

    return timesteps, signal*0.1+tanhsignal*2+noise*0.1+signal*0.1+nsignal*0.1


def sine_abate_signal(numtimesteps, time):
    timesteps = np.linspace(0, time, numtimesteps)
    stationary = time/24
    abate = np.e ** (-timesteps * stationary ** -1)
    omega = 50
    sinesignal = abate / max(abate) * np.sin(timesteps * omega)*0.4+1
    return timesteps, sinesignal


def complex_signal(numtimesteps, time):
    timesteps = np.linspace(0, time, numtimesteps)
    stationary = time/4
    tanhsignal = np.tanh(timesteps * stationary ** -1)

    timesteps = np.linspace(0, 1, numtimesteps) * time
    omega = 1000
    signal = np.sin(timesteps * omega) + 1

    mu, sigma = 0, 0.2  # mean and standard deviation
    s = np.random.normal(mu, sigma, size=numtimesteps)

    Dt = time / numtimesteps
    # dt is the timescale of the noisy signal --> emulated length scale!
    dt = time / 1000
    timescale = int(dt / Dt)
    weights = np.repeat(1.0, timescale) / timescale
    noise = np.convolve(s, weights, 'same') + 1


    timesteps = np.linspace(0, time, numtimesteps)
    stationary = time/4
    abate = np.e ** (-timesteps * stationary ** -1)
    omega = 1000
    sinesignal = abate / max(abate) * np.sin(timesteps * omega)*0.4+1

    omega = 1000
    timesteps = np.linspace(0, time, numtimesteps)
    cssignal = -3 * np.sin(timesteps*omega) / timesteps / 10
    cssignal += 1
    cssignal = np.nan_to_num(signal,0)

    return timesteps, signal*0.1+tanhsignal*2+noise*0.1+sinesignal*0.1+0.2*cssignal


def test_analize_stationarity(verbose=True):
    tight_tolerance = 1e-2
    loose_tolerance = 1
    signals = {"constant": constant_signal(1000, 1),
               "ramp": ramp_signal(1000, 1),
               "rampconst": ramptoconst_signal(10000, 1),
               "sine": sine_signal(10000, 1),
               "noise": noise_signal(10000, 1),
               "convergentsine_signal": convergentsine_signal(10000, 10),
               "tanh":tanh_signal(10000,20),
               "tanh_sin":tanh_sin_signal(10000,10),
               "tanh_sin_noise": tanh_sin_noise_signal(10000, 2.8),
               "sine_abate":sine_abate_signal(20000, 4),
               "complex":complex_signal(40000,60)
               }

    expections = {"constant": (True, (0, 0), [0,0]),
                  "ramp": (False, (0, 0), [-1,-1]),
                  "rampconst": (True, (0, 0), [0.45,0.55]),
                  "sine": (True, (0.0005, 0.0005), [0,0]),
                  "noise": (True, (0.0005, 0.0005), [0,0]),
                  "convergentsine_signal": (True, (0.0005, 0.0005), [5,5.5]),
                  "tanh":(True,(0,0),[3.5,4.5]),
                  "tanh_sin":(True,(0.0005,0.0005),[4,4.75]),
                  "tanh_sin_noise":(True,(0.0005,0.0005),[0.8,1.2]),
                  "sine_abate":(True,(0.00025,0.00025),[0.35,1]),
                  "complex":(True,(0.00025,0.0025),[20,26])
                  }

    for name, sig in signals.items():
        stationarity, scales, stationary_ts = parsed_timeseries_analysis(*sig, resolvechunks=20, verbose=verbose)

        exp_stationarity, exp_scales, exp_stationarity_ts = expections[name]
        assert stationarity == exp_stationarity, f"{name} is expected to be stationary -> {exp_stationarity}"
        assert np.allclose(scales, exp_scales, rtol=loose_tolerance), f"{name} should have a timescale"
        assert (exp_stationarity_ts[0]<=stationary_ts<=exp_stationarity_ts[1]), f"{name} is stationary at {exp_stationarity_ts} instead of computed {stationary_ts}"
        print(f"successfully tested {name}")
