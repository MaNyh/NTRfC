import numpy as np
import pandas as pd

from ntrfc.math.methods import autocorr, reldiff, return_intersection


def stationarity(timesteps, values, verbose=False):
    '''
    :param timesteps: 1D np.array()
    :param values: 1D np.array()
    :param verbose: bool
    :return: stationary_timestep ,mean_value , uncertainty
    '''

    assert len(timesteps) == len(values)
    numts = len(timesteps)

    # the next two lines define the sensivity range of the analysis.
    # mins is the lower end of the sensivity
    # maxs is the upper end of the sensivity.
    # changing these values will affect the tendency of the method to find slow or fast changes of statistical properties
    mins = 4
    maxs = 16
    doinvestigate = []

    for split in range(mins, maxs):
        splitindex = numts - numts // split
        splitlength = numts - splitindex
        # https://www.statisticshowto.com/choose-bin-sizes-statistics/
        numbins = int(1 + 3.22 * np.log(splitlength))
        _, histogram_bins = np.histogram(values, bins=numbins)

        timeseries_df = pd.DataFrame({'values': values}, index=timesteps)
        investigate = stationarity_investigation(timeseries_df, splitlength, histogram_bins)
        doinvestigate.append(investigate)

    for invest in doinvestigate:
        analize_stationarity_investigation(invest)
    stationary_guesses = [i for i in doinvestigate if i.stationary == True]
    if len(stationary_guesses) > 0:
        best_guess = np.argmin([i.msescore for i in stationary_guesses])
        return stationary_guesses[best_guess].stationary_time, stationary_guesses[best_guess].stationary_index
    else:
        return False, False


class stationarity_investigation:
    def __init__(self, timeseries_df, checksplit, histogram_bins):
        self.timeseries_df = timeseries_df
        self.window_size = checksplit
        self.histogram_bins = histogram_bins

        self.msescore = None
        self.varscore = None
        self.histscore = None
        self.arccrscore = None
        self.score = None


# Define a custom function
def hist_intersect(values, compare, bins):
    hist, _ = np.histogram(values, bins=bins)

    return return_intersection(hist, compare)


def acrr_intersect(values, compare):
    accr = autocorr(values)

    return return_intersection(accr, compare)


def analize_stationarity_investigation(investigate: stationarity_investigation):
    investigate.stationary = False
    investigate.stationary_time = np.inf
    investigate.stationary_index = None

    n_rows = investigate.timeseries_df.shape[0]
    window_size = investigate.window_size
    # Split the dataframe into two parts

    df1 = investigate.timeseries_df.iloc[:n_rows - window_size, :]
    df2 = investigate.timeseries_df.iloc[n_rows - window_size:, :]

    df2_mean = df2.mean()
    df2_vars = df2.var()
    df2_acorrs = autocorr(df2.values.T[0])
    df2_hist, _ = np.histogram(df2.values.T[0], bins=investigate.histogram_bins)

    df1_rolling = df1.rolling(window_size, min_periods=window_size)

    accuracy_limit = 0.05

    # Check if the elements in means are close to the corresponding elements in df2_mean
    means = df1_rolling.mean().values
    mean_are_close = reldiff(means.T[0], df2_mean.values[0]) < accuracy_limit
    last_false_index_mean = (~mean_are_close).argmin()
    if last_false_index_mean == 0:
        return

    vars = df1_rolling.var().values
    vars_are_close = reldiff(vars.T[0], df2_vars.values[0]) < accuracy_limit
    last_false_index_vars = (~vars_are_close).argmin()
    if last_false_index_vars == 0:
        return

    hists_intersect = df1_rolling.apply(hist_intersect, args=(df2_hist, investigate.histogram_bins,), raw=True).values
    hists_are_close = reldiff(hists_intersect.T[0], 1.0) < accuracy_limit
    last_false_index_hists = (~hists_are_close).argmin()
    if last_false_index_hists == 0:
        return

    accrs_intersect = df1_rolling.apply(acrr_intersect, args=(df2_acorrs,), raw=True).values
    accrs_are_close = reldiff(accrs_intersect.T[0], 1.0) < accuracy_limit
    last_false_index_accrs = (~accrs_are_close).argmin()
    if last_false_index_accrs == 0:
        return

    max_idx = np.max([last_false_index_mean, last_false_index_vars, last_false_index_hists, last_false_index_accrs])
    # take into account that max_id is on the right side of the interval.
    # also we need to concider that also df_1 is stationary
    id = max_idx - window_size + 1
    investigate.stationary = True
    investigate.stationary_time = investigate.timeseries_df.index[id]
    investigate.stationary_index = id

    investigate.msescore = ((df2.mean() - df1_rolling.mean()) ** 2).max()
    investigate.varscore = ((df2.var() - df1_rolling.var()) ** 2).max()
    investigate.histscore = ((1 - hists_intersect[~np.isnan(hists_intersect)]) ** 2).max()
    investigate.arccrscore = ((1 - accrs_intersect[~np.isnan(accrs_intersect)]) ** 2).max()
    investigate.score = investigate.msescore + investigate.varscore + investigate.histscore + investigate.arccrscore
