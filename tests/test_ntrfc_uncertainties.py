def test_stationarity_uncertainties_stationarysine():
    from ntrfc.postprocessing.timeseries.uncertainties import stationarity_uncertainties
    import numpy as np
    from itertools import product

    def signalgen_sine(amplitude, frequency, mean, time):
        frequency_resolution = 200
        tau = frequency ** -1

        step = tau / frequency_resolution
        times = np.arange(0, time, step)
        values = amplitude * np.sin(frequency * (2 * np.pi) * times) + mean
        return times, values

    test_amplitudes = [1, 10]
    test_frequencies = [1, 10]
    test_times = [10, 60]
    test_mean = [0, 1, -1]

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
        stationary_timestep, uncertainty = stationarity_uncertainties(timesteps, values)
        # todo: implement analytical or value-defined limits for stationarity-approximations.
        well_computed_stationarity_limit = 0.0
        well_computed_uncertainty_limit = amplitude
        assert stationary_timestep == well_computed_stationarity_limit, "computation failed"
        # todo: define uncertainty limit properly
        assert well_computed_uncertainty_limit > uncertainty, "computation failed"
    assert 10 > minperiods, f"critical value {minperiods} not tested"


def test_stationarity_uncertainties_abatingsine():
    from ntrfc.postprocessing.timeseries.uncertainties import stationarity_uncertainties
    import numpy as np
    from itertools import product

    def signalgen_abatingsine(amplitude, frequency, mean, abate, time):
        resolution = 200
        step = (resolution * frequency ** -1) ** -1
        times = np.arange(0, time, step)
        values = amplitude * np.sin(frequency * (2 * np.pi) * times) + mean + np.e ** -(times * abate)
        return times, values

    test_amplitudes = [1, 10]
    test_frequencies = [1, 10]
    test_times = [10, 60]
    test_mean = [0, 1, -1]
    test_abate = [1, 10]

    maxperiods = -1
    minperiods = np.inf

    test_configs = list(product(test_amplitudes, test_frequencies, test_times, test_mean, test_abate))

    for amplitude, frequency, time, mean, abate in test_configs:
        equals_periods = time / frequency ** -1
        if equals_periods > maxperiods:
            maxperiods = int(equals_periods)
        if minperiods > equals_periods:
            minperiods = int(equals_periods)

        timesteps, values = signalgen_abatingsine(amplitude=amplitude, frequency=frequency, mean=mean, time=time,
                                                  abate=abate)
        stationary_timestep, uncertainty = stationarity_uncertainties(timesteps, values)

        # todo: implement analytical or value-defined limits for stationarity-approximations.
        well_computed_stationarity_limit = 0.0
        well_computed_uncertainty_limit = amplitude
        assert stationary_timestep == well_computed_stationarity_limit, "computation failed"
        # todo: define uncertainty limit properly
        assert well_computed_uncertainty_limit > uncertainty, "computation failed"
    assert 10 > minperiods, f"critical value {minperiods} not tested"
