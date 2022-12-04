import numpy as np
import matplotlib.pyplot as plt

from ntrfc.math.methods import autocorr
from dataclasses import dataclass

def stationarity(timesteps, values, verbose=False):
    '''
    :param timesteps: 1D np.array()
    :param values: 1D np.array()
    :param verbose: bool
    :return: stationary_timestep ,mean_value , uncertainty
    '''

    assert len(timesteps)==len(values)
    numts = len(timesteps)
    mins = 2
    maxs = 32
    doinvestigate = []

    for split in range(mins,maxs):

        splitindex = numts-numts // split
        ref_valsplit = values[splitindex:]
        splitlength = numts-splitindex
        # https://www.statisticshowto.com/choose-bin-sizes-statistics/
        numbins = int(1+3.22*np.log(splitlength))
        _, histogram_bins = np.histogram(values,bins=numbins)

        mean_all = np.mean(ref_valsplit)
        var_ref = np.var(ref_valsplit)
        autocorr_ref = autocorr(ref_valsplit)
        hist_ref = np.histogram(ref_valsplit, bins=histogram_bins, density=True)

        investigate = investigation(mean_all, var_ref, autocorr_ref, hist_ref, splitindex, timesteps, values)

        doinvestigate.append(investigate)

    for invest in doinvestigate:
        invest.compare()
    stationary_guesses = [i for i in doinvestigate if i.stationary==True]
    if len(stationary_guesses)>0:
        best_guess = np.argmin([i.stationary_time for i in stationary_guesses])
        return stationary_guesses[best_guess].stationary_time
    else:
        return False


@dataclass
class investigation:
    mean_ref: float
    var_ref: float
    autocorr_ref: float
    hist_ref: np.histogram

    checkssplit: int

    timesteps: np.array
    values: np.array

    def compare(self):
        means = []
        varis = []
        acrrs = []
        hists = []

        self.stationary = False
        self.stationary_time = np.inf
        self.score = 1

        mean_reldiffs = []
        var_reldiffs = []
        accr_reldiffs = []
        hist_reldiffs = []

        lower = np.arange(self.checkssplit - (len(self.timesteps) - self.checkssplit), -1, -1)
        higher = np.arange(len(self.timesteps) - (len(self.timesteps) - self.checkssplit), len(self.timesteps) - self.checkssplit - 1, -1)

        limit = 0.05

        for l,h in zip(lower,higher):

            compare_vs = self.values[l:h]


            mean_all = np.mean(compare_vs)
            var_all = np.var(compare_vs)
            autocorr_all = autocorr(compare_vs)
            hist_all = np.histogram(compare_vs, bins=self.hist_ref[1], density=True)

            means.append(mean_all)
            varis.append(var_all)
            acrrs.append(autocorr_all)
            hists.append(hist_all)

            mean_reldiffs.append(reldiff(self.mean_ref, mean_all))
            var_reldiffs.append(reldiff(self.var_ref, var_all))
            accr_reldiffs.append(1-return_intersection(autocorr_all,self.autocorr_ref))

            #https://mpatacchiola.github.io/blog/2016/11/12/the-simplest-classifier-histogram-intersection.html
            histdiff =[1 - return_intersection( _a, _b) for _a, _b in zip(self.hist_ref[0], hist_all[0])]

            meanhistdiff = np.nanmean(histdiff)
            hist_reldiffs.append(meanhistdiff)

            accuracylimit_reached = limit>np.max(mean_reldiffs) and limit>np.max(var_reldiffs) and limit>np.max(accr_reldiffs) and limit>np.max(hist_reldiffs)
            if accuracylimit_reached and self.timesteps[l]<self.stationary_time:
                self.stationary = True
                self.stationary_time = self.timesteps[l]
                self.score = np.nanmean([mean_reldiffs,var_reldiffs,accr_reldiffs,hist_reldiffs])
            elif not accuracylimit_reached:
                break




def reldiff(a,b):
    #https://en.wikipedia.org/wiki/Relative_change_and_difference
    if a==b:
        return 0
    else:
        return abs(abs(a)-abs(b))/((abs(a)+abs(b))/2)

def return_intersection(hist_1, hist_2):
    minima = np.minimum(hist_1, hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection
