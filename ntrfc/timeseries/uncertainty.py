import numpy as np
import matplotlib.pyplot as plt

from ntrfc.timeseries.integral_scales import integralscales
from dataclasses import dataclass

def stationarity_uncertainties(timesteps, values, verbose=False):
    '''
    :param timesteps: 1D np.array()
    :param values: 1D np.array()
    :param verbose: bool
    :return: stationary_timestep ,mean_value , uncertainty
    '''

    assert len(timesteps)==len(values)
    numts = len(timesteps)
    mins = 8
    maxs = 100
    idealsplits = int(numts/256)
    maxsplits = mins if idealsplits < mins else maxs if idealsplits > maxs else idealsplits
    check_splits = range(mins,maxsplits)
    doinvestigate = []

    for split in check_splits:
        timesplits = np.array_split(timesteps,split)
        valsplit = np.array_split(values,split)


        for cs,(ts,vs) in enumerate(zip(timesplits,valsplit)):
            # https://www.statisticshowto.com/choose-bin-sizes-statistics/
            numbins = int(1+3.22*np.log(len(ts)))
            _, histogram_bins = np.histogram(values,bins=numbins)

            mean_all = np.mean(vs)
            var_all = np.var(vs)
            autocorr_all,lengthscal_all = integralscales(vs,ts)
            hist_all = np.histogram(vs,bins=histogram_bins,density=True)

            investigate = investigation(mean_all,var_all,autocorr_all,hist_all,cs ,timesplits,valsplit)

            doinvestigate.append(investigate)

    for invest in doinvestigate:
        invest.compare()
    stationary_guesses = [i for i in doinvestigate if i.stationary==True]
    best_guess = np.argmin([i.stationary_time for i in stationary_guesses])
    return stationary_guesses[best_guess].stationary_time

@dataclass
class investigation:
    mean_all: float
    var_all: float
    autocorr_all: float
    hist_all: np.histogram

    checkssplit: int

    timesplits: np.array
    valsplits: np.array

    def plot_tobeinvestigated(self):
        plt.figure()
        for ts,vs in zip(self.timesplits,self.valsplits):
            plt.plot(ts,vs)
            plt.vlines(ts[-1],ymin=np.min(self.valsplits[0]),ymax=np.max(self.valsplits[0]))
            if self.stationary_time:
                plt.axvline(self.stationary_time)
        plt.show()

    def compare(self):
        means = []
        varis = []
        acrrs = []
        hists = []

        valid_timesplits=self.timesplits[:]
        del valid_timesplits[self.checkssplit]
        valid_valsplits=self.valsplits[:]
        del valid_valsplits[self.checkssplit]


        for cs ,(ts,vs) in enumerate(zip(valid_timesplits,valid_valsplits)):
            mean_all = np.mean(vs)
            var_all = np.var(vs)
            autocorr_all, lengthscal_all= integralscales(vs,ts)
            hist_all = np.histogram(vs,bins=self.hist_all[1],density=True)
            means.append(mean_all)
            varis.append(var_all)
            acrrs.append(autocorr_all)
            hists.append(hist_all)

        mean_reldiffs = []
        var_reldiffs = []
        accr_reldiffs = []
        hist_reldiffs = []

        for mean, vari, accr, hist in zip(means, varis,acrrs,hists):
            mean_reldiffs.append(reldiff(self.mean_all,mean))
            var_reldiffs.append(reldiff(self.var_all,vari))
            accr_reldiffs.append(reldiff(self.autocorr_all,accr))

            #https://mpatacchiola.github.io/blog/2016/11/12/the-simplest-classifier-histogram-intersection.html
            histdiff =[1-return_intersection( _a, _b) for _a, _b in zip(self.hist_all[0], hist[0])]

            meanhistdiff = np.nanmean(histdiff)
            hist_reldiffs.append(meanhistdiff)

        num_splits = len(self.timesplits)
        self.mean_rdmeans = [np.mean(mean_reldiffs[i:]) for i in range(num_splits-1)]
        self.var_rdmeans = [np.mean(var_reldiffs[i:]) for i in range(num_splits-1)]
        self.accr_rdmeans = [np.mean(accr_reldiffs[i:]) for i in range(num_splits-1)]
        self.hist_rdmeans = [np.mean(hist_reldiffs[i:]) for i in range(num_splits-1)]



        if 0.05>np.nanmin(self.mean_rdmeans) and 0.05>np.nanmin(self.var_rdmeans) and 0.05>np.nanmin(self.accr_rdmeans) and 0.05>np.nanmin(self.hist_rdmeans):

            self.stationary = True
            self.scores = [np.nanmean([self.mean_rdmeans[i],self.var_rdmeans[i],self.accr_rdmeans[i],self.hist_rdmeans[i]]) for i in range(num_splits-1)]
            best = np.nanargmin(self.scores)
            self.score = self.scores[best]
            self.stationary_time=self.timesplits[best][0]
        else:
            self.stationary = False



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
