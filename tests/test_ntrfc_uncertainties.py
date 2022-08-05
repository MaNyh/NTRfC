

def test_stationarity_uncertainties():
    from ntrfc.postprocessing.timeseries.uncertainties import stationarity_uncertainties
    import numpy as np

    timesteps = np.arange(0,60,0.001)
    values = np.sin(timesteps)

    ans = stationarity_uncertainties(timesteps,values)
    return 0
