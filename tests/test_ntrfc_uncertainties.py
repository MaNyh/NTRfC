
def test_stationarity_uncertainties():
    from ntrfc.postprocessing.timeseries.uncertainties import stationarity_uncertainties
    import numpy as np
    def gen_sine():
        amplitude = 1
        time = 60
        frequency = 2
        timesteps = np.arange(0, time, 0.001)
        values = amplitude * np.sin(frequency * timesteps)
        return timesteps, values
    timesteps,values= gen_sine()

    ans = stationarity_uncertainties(timesteps,values)

    return 0



