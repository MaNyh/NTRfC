import numpy as np
import matplotlib.pyplot as plt
import bisect
from scipy.integrate import simps

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.stats







def stationarity_uncertainties(timesteps, values, verbose=False):
    '''
    :param timesteps: 1D np.array()
    :param values: 1D np.array()
    :param verbose: bool
    :return: stationary_timestep ,mean_value , uncertainty
    '''

    ### commented is for rms

    # confidence_mean_high = [];  # upper limit
    # confidence_mean_low = [];  # lower limit
    # mean_in_ci_range = [];  # is value in confidence range?
    # out_of_range = [];

    # we are integrating from zero to zero-crossing in the autocorrelation, we need the time to begin with zeros
    # probably the used datasample is not beginning with 0. therefore:

    time = timesteps

    somesignal = values

    error = 0.00005

    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
        return m, m - h, m + h

    def statistical_timewindow(somesignal, time, error):
        numtimesteps = len(time)
        dt = time[1] - time[0]

        minobservations_in_window = int(len(somesignal) / 100) if int(len(somesignal) / 100) > 256 else 256  # 256
        jumpsteps = 1
        windows_idxs = [numtimesteps - i for i in np.arange(minobservations_in_window, numtimesteps, jumpsteps)]

        pspecs = []
        times = []
        for check_id in windows_idxs:
            signal_window = somesignal[check_id:]
            time_window = time[check_id:]

            fs = dt ** -1
            f, Pxx_spec = signal.welch(signal_window, fs, 'flattop', scaling='spectrum')
            pspecs.append(Pxx_spec)
            times.append(time_window[0])
            if len(pspecs) > 2:
                pspec_diff_sqr = np.abs(pspecs[-2] - pspecs[-1]) / pspecs[
                    -1]  # **2/(pspecs[-2]*pspecs[-1])#/(times[-1]-times[2])
                pspecs_diff_sqrint = np.trapz(pspec_diff_sqr, f) / np.trapz(pspecs[-1], f)  # ,fr)/(times[-2]-times[-1])
                if pspecs_diff_sqrint < error:
                    # statistical_convergence_timestep = check_id

                    return check_id, pspecs, times

        return -1, -1, -1

    stationarity_window_timestep, pspecs, times = statistical_timewindow(somesignal, time, error)
    print(stationarity_window_timestep)
    n = len(time)
    sample_length = n - stationarity_window_timestep

    plt.figure()
    plt.hist(somesignal[stationarity_window_timestep:], bins=100)
    plt.show()

    plt.figure()
    plt.plot(somesignal[stationarity_window_timestep:])
    plt.show()

    samplesignal = somesignal[stationarity_window_timestep:]
    sampletime = time[stationarity_window_timestep:]
    dt = time[1] - time[0]

    stationary = [stationarity_window_timestep, n]
    checkstationary = [0, stationarity_window_timestep - 1]
    nonstationary = [None, None]

    searchbool = True
    new_sample_idx = stationary[0]
    new_sample_idxbegin = stationary[0]-sample_length

    newsamplesignal = somesignal[new_sample_idxbegin:new_sample_idx]
    newsampletime = time[new_sample_idxbegin:new_sample_idx]
    # while(searchbool):
    fs = dt ** -1
    f, Pxx_spec = signal.welch(newsamplesignal, fs, 'flattop', scaling='spectrum')

    pspec_diff_sqr = np.abs(Pxx_spec - pspecs[-1]) / pspecs[-1]  # **2/(pspecs[-2]*pspecs[-1])#/(times[-1]-times[2])
    pspecs_diff_sqrint = np.trapz(pspec_diff_sqr, f) / np.trapz(pspecs[-1], f)  # ,fr)/(times[-2]-times[-1])
    nerr = (error + error) / (1 - 2 * error) ** 2
    nnerr = 2 * nerr
    print(pspecs_diff_sqrint < nnerr)
    print(pspecs_diff_sqrint)

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
    plt.plot(newsampletime, newsamplesignal, color="orange")
    plt.axvline(time[new_sample_idxbegin], color="black", linestyle="dashed")
    plt.axvline(time[new_sample_idx], color="black", linestyle="dashed")
    plt.ylim(min(somesignal), max(somesignal))
    plt.show()

    id_stationary=new_sample_idxbegin

    return timesteps[id_stationary], time[new_sample_idxbegin]



