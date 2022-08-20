import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from ntrfc.postprocessing.timeseries.integral_scales import integralscales

def minimal_statistical_timewindow(somesignal, time, error):
    """
    computes the minimal statistical time window for a given error in the energy-spectrum of a signal
    if the energy-spectrum is converged, a necessary condition for stationary is satisfied

    """
    numtimesteps = len(time)
    dt = time[1] - time[0]

    minobservations_in_window = int(len(somesignal) / 100) if int(len(somesignal) / 100) > 256 else 256  # 256
    jumpsteps = 1
    windows_idxs = [numtimesteps - i for i in np.arange(minobservations_in_window, numtimesteps, jumpsteps)]

    pspecs = []
    times = []
    for check_id in windows_idxs:
        # choosing window of the signal. starting from end, iterate to a longer signal towards beginning
        signal_window = somesignal[check_id:]
        time_window = time[check_id:]

        # compute power spectrum
        fs = dt ** -1
        f, Pxx_spec = signal.welch(signal_window, fs, 'flattop', scaling='spectrum')
        pspecs.append(Pxx_spec)
        times.append(time_window[0])

        if len(pspecs) > 2:
            pspec_diff_sqr = np.abs(pspecs[-2] - pspecs[-1]) / pspecs[-1]
            pspecs_diff_sqrint = np.trapz(pspec_diff_sqr, f) / np.trapz(pspecs[-1], f)

            if pspecs_diff_sqrint < error:
                return check_id, pspecs, times

    return -1, -1, -1


def stationarity_uncertainties(timesteps, values, verbose=True):
    '''
    :param timesteps: 1D np.array()
    :param values: 1D np.array()
    :param verbose: bool
    :return: stationary_timestep ,mean_value , uncertainty
    '''

    time = timesteps
    somesignal = values

    error = 0.01
    error_sec = 0.05
    dt = time[1] - time[0]
    n = len(time)

    # compute minimal time window for the approximation of a stationary signal
    minimal_stationarity_timestep, pspecs, times = minimal_statistical_timewindow(somesignal, time, error)


    error_sec_test = 0
    checks = 0
    sample_start = minimal_stationarity_timestep
    new_sample_idxbegin = n
    means = []
    spectralerrors = []
    vars = []
    integralscale= []
    statpairids = []
    while (error_sec > error_sec_test and sample_start > 0):
        stationary = [sample_start, n - checks]

        sampletime = time[stationary[0]:stationary[-1]]

        newsamplesignal = somesignal[stationary[0]:stationary[-1]]

        fs = dt ** -1
        f, Pxx_spec = signal.welch(newsamplesignal, fs, 'flattop', scaling='spectrum')

        pspec_diff_sqr = np.abs(Pxx_spec - pspecs[-1]) / pspecs[-1]
        pspecs_diff_sqrint = np.trapz(pspec_diff_sqr, f) / np.trapz(pspecs[-1], f)
        error_sec_test = pspecs_diff_sqrint
        spectralerrors.append(error_sec_test)
        means.append(np.mean(newsamplesignal))
        vars.append(np.var(newsamplesignal))
        tau, lenghtscale = integralscales(newsamplesignal,sampletime)
        integralscale.append(tau)
        statpairids.append(stationary)
        checks += 1
        sample_start -= 1

    id_stationary = new_sample_idxbegin - new_sample_idxbegin
    uncertainty = np.std(means)
    if verbose:
        plt.figure()
        plt.hist(somesignal[minimal_stationarity_timestep:], bins=100)
        plt.show()

        plt.figure()
        plt.plot(somesignal[minimal_stationarity_timestep:])
        plt.show()

        plt.figure()
        plt.plot(f, Pxx_spec)
        plt.plot(f, pspecs[-1])
        plt.yscale("log")
        plt.xscale("log")
        plt.plot()

        plt.figure()
        plt.plot(pspec_diff_sqr)
        plt.yscale("log")
        plt.xscale("log")
        plt.plot()

        plt.figure()
        plt.plot(time, somesignal)
        plt.axvline(sampletime[-1], color="red")
        plt.axvline(sampletime[0], color="red")
        plt.plot(time[id_stationary:], somesignal[id_stationary:], color="orange")
        plt.axvline(time[id_stationary], color="black", linestyle="dashed")
        plt.ylim(min(somesignal), max(somesignal))
        plt.show()

    return timesteps[id_stationary], uncertainty
