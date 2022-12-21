from dataclasses import dataclass

import numpy as np

from ntrfc.math.methods import autocorr


def stationarity(timesteps, values, verbose=False):
    '''
    :param timesteps: 1D np.array()
    :param values: 1D np.array()
    :param verbose: bool
    :return: stationary_timestep ,mean_value , uncertainty
    '''

    assert len(timesteps) == len(values)
    numts = len(timesteps)
    mins = 2
    maxs = 32
    doinvestigate = []

    for split in range(mins, maxs):
        splitindex = numts - numts // split
        ref_valsplit = values[splitindex:]
        splitlength = numts - splitindex
        # https://www.statisticshowto.com/choose-bin-sizes-statistics/
        numbins = int(1 + 3.22 * np.log(splitlength))
        _, histogram_bins = np.histogram(values, bins=numbins)

        mean_all = np.mean(ref_valsplit)
        var_ref = np.var(ref_valsplit)
        autocorr_ref = autocorr(ref_valsplit)
        hist_ref = np.histogram(ref_valsplit, bins=histogram_bins, density=True)

        investigate = stationarity_investigation(mean_all, var_ref, autocorr_ref, hist_ref, splitindex, timesteps,
                                                 values)

        doinvestigate.append(investigate)

    for invest in doinvestigate:
        analize_stationarity_investigation(invest)
    stationary_guesses = [i for i in doinvestigate if i.stationary == True]
    if len(stationary_guesses) > 0:
        best_guess = np.argmin([i.stationary_time for i in stationary_guesses])
        return stationary_guesses[best_guess].stationary_time
    else:
        return False


@dataclass
class stationarity_investigation:
    mean_ref: float
    var_ref: float
    autocorr_ref: float
    hist_ref: np.histogram

    checkssplit: int

    timesteps: np.array
    values: np.array


def analize_stationarity_investigation(investigate: stationarity_investigation):
    means = []
    varis = []
    acrrs = []
    hists = []

    investigate.stationary = False
    investigate.stationary_time = np.inf
    investigate.score = 1

    mean_reldiffs = []
    var_reldiffs = []
    accr_reldiffs = []
    hist_reldiffs = []

    lower_íds = np.arange(investigate.checkssplit - (len(investigate.timesteps) - investigate.checkssplit), -1, -1)
    higher_ids = np.arange(len(investigate.timesteps) - (len(investigate.timesteps) - investigate.checkssplit),
                           len(investigate.timesteps) - investigate.checkssplit - 1, -1)

    accuracy_limit = 0.05

    for l, h in zip(lower_íds, higher_ids):

        compare_vs = investigate.values[l:h]

        mean_all = np.mean(compare_vs)
        var_all = np.var(compare_vs)
        autocorr_all = autocorr(compare_vs)
        hist_all = np.histogram(compare_vs, bins=investigate.hist_ref[1], density=True)

        means.append(mean_all)
        varis.append(var_all)
        acrrs.append(autocorr_all)
        hists.append(hist_all)

        mean_reldiffs.append(reldiff(investigate.mean_ref, mean_all))
        var_reldiffs.append(reldiff(investigate.var_ref, var_all))
        accr_reldiffs.append(1 - return_intersection(autocorr_all, investigate.autocorr_ref))

        # https://mpatacchiola.github.io/blog/2016/11/12/the-simplest-classifier-histogram-intersection.html
        histdiff = [1 - return_intersection(_a, _b) for _a, _b in zip(investigate.hist_ref[0], hist_all[0])]

        meanhistdiff = np.nanmean(histdiff)
        hist_reldiffs.append(meanhistdiff)

        accuracylimit_reached = accuracy_limit > np.max(mean_reldiffs) and accuracy_limit > np.max(
            var_reldiffs) and accuracy_limit > np.max(accr_reldiffs) and accuracy_limit > np.max(hist_reldiffs)
        if accuracylimit_reached and investigate.timesteps[l] < investigate.stationary_time:
            investigate.stationary = True
            investigate.stationary_time = investigate.timesteps[l]
            investigate.score = np.nanmean([mean_reldiffs, var_reldiffs, accr_reldiffs, hist_reldiffs])
        elif not accuracylimit_reached:
            break


def reldiff(a, b):
    # https://en.wikipedia.org/wiki/Relative_change_and_difference
    if a == b:
        return 0
    else:
        return abs(abs(a) - abs(b)) / ((abs(a) + abs(b)) / 2)


def return_intersection(hist_1, hist_2):
    minima = np.minimum(hist_1, hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection
