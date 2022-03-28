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
        csig=signal
        ctime=timesteps
        plot_stationarity_analisys(csig, ctime, signal, stationarity_timestep, timesteps)

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

            plot_stationarity_analisys(csig, ctime, signal, stationarity_timestep, timesteps)

            return stationarity, scales, stationarity_timestep

    plot_stationarity_analisys(csig, ctime, signal, stationarity_timestep, timesteps)
    return stationarity, scales, stationarity_timestep


def plot_stationarity_analisys(csig, ctime, signal, stationarity_timestep, timesteps):
    plt.figure()
    plt.plot(timesteps, signal)
    plt.plot(ctime, csig, color="black", linewidth=4)
    plt.xlim(0, timesteps[-1])
    sts = stationarity_timestep if stationarity_timestep>=0 else None
    ymin,ymax = min(signal),max(signal)
    if ymax-ymin<0.01:
        ymax=0.5+np.mean(signal)
        ymin =-0.5+np.mean(signal)
    if sts==0.0 or sts:
        plt.vlines(sts, ymin=ymin, ymax=ymax,
                   linewidth=4, color="k", linestyles="dashed")
        plt.axvspan(sts, timesteps[-1],
                    facecolor='green', alpha=0.5)
    else:
        plt.axvspan(0, timesteps[-1],
                    facecolor='red', alpha=0.5)

    plt.ylim(ymin, ymax)
    plt.show()  #


def check_signal_stationarity(resolvechunks, signal, timesteps, verbose = True):

    checksigchunks = chunks(signal, int(len(signal) / resolvechunks))
    checktimechunks = chunks(timesteps, int(len(timesteps) / resolvechunks))

    # a new approach could be to check chunkwise the stationarity by logic.
    # a constant has a constant mean but no trend and no variation
    # a trend has a constant trend, but no mean and no variation
    # a correlating signal has a time and length scale, a mean, a constant variation and autocorrelation

    mean = np.mean(signal)
    means = np.array([np.mean(i) for i in checksigchunks]) #np.mean(checksigchunks, axis=1)

    var = np.std(signal)
    vars = np.array([np.std(i) for i in checksigchunks])#np.std(checksigchunks, axis=1)

    # todo: now it is only mean, val and var that is being investigated.
    # it makes sense to also investigate the behaviour of the autocorrelation
    # but as the signal is divided into chunks, one has to
    const_mean = np.allclose(mean, means,rtol=0.05)
    const_val = np.allclose(mean, signal,rtol=0.05)
    const_var = np.allclose(var,vars,rtol=0.4)
    #
    if const_mean and const_var:
        timescale, lengthscale = integralscales(signal, timesteps)
        timescales, lengthscales = zip(*[integralscales(s, t) for s, t in zip(checksigchunks, checktimechunks)])
        # as in ries2018, a scale can only be computed when enough scales are within the signal
        if (timesteps[-1]-timesteps[0])/timescale<30:
            print("warning")
    else:
        timescale, lengthscale = 0,0
        timescales, lengthscales = np.zeros(resolvechunks),np.zeros(resolvechunks)

    const_tscales = np.allclose(timescales, timescale,rtol=0.1)

    """
    from itertools import product
    const_mean_allowed = [{"const_mean":True}]
    const_val_allowed = [{"const_val":True},{"const_val":False}]
    const_var_allowed = [{"const_var":True},{"const_var":False}]
    const_tscale_allowed = [{"const_tscale":True},{"const_tscale":False}]

    combinations = list(product(const_mean_allowed,const_var_allowed,const_val_allowed,const_tscale_allowed))
    """

    if const_mean and const_var and const_val and const_tscales:
        # constant. scales are computed but not valid
        signal_type = "constant"
        stationarity = True
        stationarity_timestep = timesteps[0]
    elif const_mean and const_var and const_val and not const_tscales:
        # constant. scales are not computed
        signal_type = "constant"
        stationarity = True
        stationarity_timestep = timesteps[0]
    elif const_mean and const_var and not const_val and const_tscales:
        # selfcorrelating stationary
        signal_type = "selfcorrelating stationary"
        stationarity = True
        stationarity_timestep = timesteps[0]
    elif const_mean and const_var and not const_val and not const_tscales:
        # weak stationary
        signal_type = "weak stationary"
        stationarity = True
        stationarity_timestep = timesteps[0]
    elif const_mean and not const_var and const_val and const_tscales:
        # weak stationary
        signal_type = "weak stationary"
        stationarity = True
        stationarity_timestep = timesteps[0]
    elif const_mean and not const_var and const_val and not const_tscales:
        # weak stationary
        signal_type = "weak constant"
        stationarity = True
        stationarity_timestep = timesteps[0]
    # elif const_mean and not const_var and not const_val and const_tscales:
    #     # weak stationary
    #     signal_type = "nonstationary"
    #     stationarity = False
    #     stationarity_timestep = timesteps[0]
    # elif const_mean and not const_var and not const_val and not const_tscales:
    #     # weak stationary
    #     signal_type = nonstationary"
    #     stationarity = False
    #     stationarity_timestep = -1

    else:
        # nonstationary signal
        signal_type = "nonstationary"
        stationarity = False
        stationarity_timestep = -1


    return signal_type, stationarity, stationarity_timestep, timescale, lengthscale
