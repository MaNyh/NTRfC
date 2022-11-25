import os
import pytest
ON_CI = 'CI' in os.environ

def test_stationarity_uncertainties_stationarysine(verbose=False):
    from ntrfc.timeseries.uncertainty import stationarity_uncertainties
    import numpy as np
    from itertools import product
    import matplotlib.pyplot as plt

    def signalgen_sine(amplitude, frequency, mean, time):
        frequency_resolution = 512
        tau = frequency ** -1

        step = tau / frequency_resolution
        times = np.arange(0, time, step)
        values = amplitude * np.sin(frequency * (2 * np.pi) * times) + mean
        return times, values


    test_amplitudes = [1] #1,10
    test_frequencies = [1]#1,10
    test_times = [10] #10, 60
    test_mean = [-1]#-1,0,1

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
        stationary_timestep = stationarity_uncertainties(timesteps, values)
        analytic_stationary_limit = 0.0
        assert stationary_timestep == analytic_stationary_limit, "computation failed"

        if verbose:
            plt.figure()
            plt.plot(timesteps,values)
            plt.axvline(stationary_timestep)


def test_stationarity_uncertainties_abatingsine(verbose=False):
    from ntrfc.timeseries.uncertainty import stationarity_uncertainties
    import numpy as np
    from itertools import product
    import matplotlib.pyplot as plt

    def signalgen_abatingsine(amplitude, frequency, mean, abate, time):
        resolution = 2048
        step = (resolution * frequency ** -1) ** -1
        times = np.arange(0, time, step)
        values = amplitude * np.sin(frequency * (2 * np.pi) * times) + mean + np.e ** -(times * abate)
        return times, values

    test_amplitudes = [0.2]
    test_frequencies = [10]
    test_times = [20]
    test_mean = [-1]
    test_abate = [1, 10]


    test_configs = list(product(test_amplitudes, test_frequencies, test_times, test_mean, test_abate))

    for amplitude, frequency, time, mean, abate in test_configs:


        timesteps, values = signalgen_abatingsine(amplitude=amplitude, frequency=frequency, mean=mean, time=time,
                                                  abate=abate)
        stationary_timestep = stationarity_uncertainties(timesteps, values)

        well_computed_stationarity_limit = -np.log(0.05) / abate
        if verbose:
            plt.figure()
            plt.plot(timesteps,values)
            plt.axvline(stationary_timestep)
            plt.show()
        assert  stationary_timestep >well_computed_stationarity_limit , "computation failed"


def test_stationarity_uncertainties_abatingsinenoise(verbose=False):
    from ntrfc.timeseries.uncertainty import stationarity_uncertainties
    import numpy as np
    from itertools import product
    import matplotlib.pyplot as plt

    def signalgen_abatingsine(amplitude,noiseamplitude, frequency, mean, abate, time):
        resolution = 2048
        step = (resolution * frequency ** -1) ** -1

        times = np.arange(0, time, step)
        noise = np.random.normal(-1,1,len(times))*noiseamplitude

        values = amplitude * np.sin(frequency * (2 * np.pi) * times) + mean + np.e ** -(times * abate)+noise
        return times, values

    test_amplitudes = [2]
    test_noiseamplitude = [0.1]
    test_frequencies = [6]
    test_times = [20]
    test_mean = [-1]
    test_abate = [0.5]


    test_configs = list(product(test_amplitudes,test_noiseamplitude, test_frequencies, test_times, test_mean, test_abate))

    for amplitude, noiseamplitude, frequency, time, mean, abate in test_configs:


        timesteps, values = signalgen_abatingsine(amplitude=amplitude, noiseamplitude=noiseamplitude,frequency=frequency, mean=mean, time=time,
                                                  abate=abate)
        stationary_timestep = stationarity_uncertainties(timesteps, values)

        well_computed_stationarity_limit = -np.log(0.05) / abate
        if verbose:
            plt.figure()
            plt.plot(timesteps,values)
            plt.axvline(stationary_timestep)
            plt.show()
        assert 3*well_computed_stationarity_limit > stationary_timestep > well_computed_stationarity_limit , "computation failed"

@pytest.mark.skipif(ON_CI, reason="currently cant render on ci-systems")
def test_stationarity_uncertainties_abating(verbose=False):
    from ntrfc.timeseries.uncertainty import stationarity_uncertainties
    import numpy as np
    from itertools import product
    import matplotlib.pyplot as plt

    def signalgen_abatingsine(noiseamplitude,  mean, abate, time):
        resolution = 30000
        step = (time/resolution )

        times = np.arange(0, time, step)
        noise = np.random.normal(-1,1,len(times))*noiseamplitude

        values = mean + np.e ** -(times * abate)+noise
        return times, values

    test_noiseamplitude = [0.01]
    test_times = [10]
    test_mean = [-1]
    test_abate = [3]


    test_configs = list(product(test_noiseamplitude, test_times, test_mean, test_abate))

    for noiseamplitude,  time, mean, abate in test_configs:


        timesteps, values = signalgen_abatingsine(noiseamplitude=noiseamplitude, mean=mean, time=time,
                                                  abate=abate)
        stationary_timestep = stationarity_uncertainties(timesteps, values)

        well_computed_stationarity_limit = -np.log(0.05) / abate
        if verbose:
            plt.figure()
            plt.plot(timesteps,values)
            plt.axvline(stationary_timestep)
            plt.show()
        assert 3*well_computed_stationarity_limit > stationary_timestep >well_computed_stationarity_limit , "computation failed"


