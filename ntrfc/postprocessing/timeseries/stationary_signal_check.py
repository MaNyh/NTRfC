import numpy as np
from matplotlib import pyplot as plt

from ntrfc.postprocessing.timeseries.integral_scales import integralscales


def chunks(somelist, numchunks):
    def split(list_a, chunk_size):
        for i in range(0, len(list_a), chunk_size):
            yield list_a[i:i + chunk_size]

    return list(split(somelist, numchunks))


def parsed_timeseries_analysis(timesteps, signal, resolvechunks=20, verbose=True):
    checksigchunks = chunks(signal, int(len(signal) / resolvechunks))
    checktimechunks = chunks(timesteps, int(len(timesteps) / resolvechunks))

    min_chunk = int(resolvechunks / 2)

    signal_type, stationarity, stationarity_timestep, timescale, lengthscale = check_signal_stationarity(resolvechunks, signal, timesteps)
    scales = (timescale,lengthscale)
    if stationarity==True:
        plt.figure()
        plt.plot(timesteps, signal, color="k" )
        plt.vlines(stationarity_timestep, ymin=0, ymax=2, linewidth=4, color="k", linestyles="dashed")
        plt.xlim(0, timesteps[-1])
        plt.ylim(-10, 10)
        plt.show()  #
        return stationarity, timescale, stationarity_timestep

    for i in range(min_chunk, resolvechunks + 1):
        csig = np.concatenate([*checksigchunks[resolvechunks - i:]])
        ctime = np.concatenate([*checktimechunks[resolvechunks - i:]])

        signal_type,newstationarity,newstationarity_timestep, newtimescale,newlengthscale = check_signal_stationarity(resolvechunks, csig, ctime)

        if newstationarity:
            newscales = (timescale, lengthscale)
            stationarity=True
            scales = newscales
            stationarity_timestep = newstationarity_timestep


        if not newstationarity:
            # when no further stationarity found, return status
            # when done, return last status
            if verbose:
                plt.figure()
                plt.plot(timesteps, signal)
                plt.plot(ctime, csig, color="black", linewidth=0.1)
                plt.vlines(stationarity_timestep, ymin=-10, ymax=10, linewidth=4, color="k", linestyles="dashed")
                plt.axvspan(stationarity_timestep, max(timesteps), facecolor='grey', alpha=0.5)

                plt.xlim(0, timesteps[-1])
                plt.ylim(-10, 10)
                plt.show()  #
            return stationarity, scales, stationarity_timestep

    plt.figure()
    plt.plot(timesteps, signal)
    plt.plot(ctime, csig, color="black", linewidth=4)
    plt.vlines(stationarity_timestep, ymin=0, ymax=2, linewidth=4, color="k", linestyles="dashed")
    plt.xlim(0, timesteps[-1])
    plt.ylim(-10, 10)
    plt.show()  #
    return stationarity, scales, stationarity_timestep


def check_signal_stationarity(resolvechunks, signal, timesteps, verbose = True):

    checksigchunks = chunks(signal, int(len(signal) / resolvechunks))
    checktimechunks = chunks(timesteps, int(len(timesteps) / resolvechunks))

    # a new approach could be to check chunkwise the stationarity by logic.
    # a constant has a constant mean but no trend and no variation
    # a trend has a constant trend, but no mean and no variation
    # a correlating signal has a time and length scale, a mean, a constant variation and autocorrelation

    mean = np.mean(signal)
    means = np.mean(checksigchunks, axis=1)

    var = np.std(signal)
    vars = np.std(checksigchunks, axis=1)

    sigma = var**.5
    tolerance = var

    const_mean = np.allclose(mean, means,rtol=0.08)

    const_val = np.allclose(mean, signal,rtol=0.08)

    const_var = np.allclose(var,vars,rtol=3)

    #
    if const_mean and const_var:
        timescale, lengthscale = integralscales(signal, timesteps)
        timescales, lengthscales = zip(*[integralscales(s, t) for s, t in zip(checksigchunks, checktimechunks)])

        if (timesteps[-1]-timesteps[0])/timescale<30:
            print("warning")
    else:
        timescale, lengthscale = 0,0
        timescales, lengthscales = np.zeros(resolvechunks),np.zeros(resolvechunks)

    const_tscales = np.allclose(timescales, timescale,rtol=tolerance)
    const_lscales = np.allclose(lengthscales, lengthscale,rtol=tolerance)

    if const_mean == const_val == const_var == True:
        # constant signal
        signal_type = "constant"
        stationarity = True
        stationarity_timestep = timesteps[0]
    elif const_mean == const_var == True and const_val == False:
        # stationary signal
        signal_type = "stationary structured"
        stationarity = True
        stationarity_timestep = timesteps[0]
    elif const_mean == True and (const_var == True or const_val == True):
        # stationary signal
        signal_type = "stationary unstructured"
        stationarity = True
        stationarity_timestep = timesteps[0]
    elif const_var == const_val == True and const_mean == False:
        # nonstationary signal
        signal_type = "nonstationary"
        stationarity = False
        stationarity_timestep = -1
    elif const_var == True and const_val == False and const_mean == False:
        # nonstationary signal
        signal_type = "nonstationary"
        stationarity = False
        stationarity_timestep = -1
    elif const_var == False and const_val == True and const_mean == False:
        # nonstationary signal
        signal_type = "nonstationary"
        stationarity = False
        stationarity_timestep = -1
    elif const_var == False and const_val == False and const_mean == False:
        # nonstationary signal
        signal_type = "nonstationary"
        stationarity = False
        stationarity_timestep = -1
    else:
        # nonstationary signal
        signal_type = "nonstationary"
        stationarity = False
        stationarity_timestep = -1


    return signal_type, stationarity, stationarity_timestep, timescale, lengthscale
