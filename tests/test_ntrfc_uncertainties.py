
def test_stationarity_uncertainties():
    from ntrfc.postprocessing.timeseries.uncertainties import stationarity_uncertainties
    import numpy as np

    def signalgen_sine(amplitude,frequency,time):
        resolution = 200
        step = (resolution*frequency**-1)**-1
        times = np.arange(0, time, step)
        values = amplitude * np.sin(frequency*(2*np.pi) * times)
        return times, values

    timesteps, values = signalgen_sine(amplitude=1, frequency=1, time=2)

    ans = stationarity_uncertainties(timesteps,values)

    return 0



