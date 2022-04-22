import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats


from ntrfc.utils.math.methods import autocorr, zero_crossings


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m-h, m+h, m

wake = np.load("wake.npz")

time = wake.f.time

signal = wake.f.pressure[:,12]

dt = time[1]-time[0]


second_half_timestepid = int(len(time)/2)

second_time = time[second_half_timestepid:]
second_signal = signal[second_half_timestepid:]
second_fluc = second_signal-np.mean(second_signal)

afc=autocorr(second_signal-np.mean(second_signal))
zeros_afc = zero_crossings(afc)
tau = zeros_afc[1]-zeros_afc[0]

scales = 8
T = scales*tau
n = len(time)
notimewindows = int((n)/T)

slides =  [signal[i*T:(i+1)*T] for i in range(notimewindows)]


sliding_vars=[np.var(i) for i in slides]
sliding_means=[np.mean(i) for i in slides]
sliding_afc = [autocorr(i-np.mean(i)) for i in slides]
sliding_zerocrossings_afcs = [zero_crossings(i) for i in sliding_afc]
sliding_T = [(i[1]-i[0])*dt for i in sliding_zerocrossings_afcs ]


vars_for_confidence = sliding_vars[int(len(sliding_vars)/2):]
vars_conf = -2*np.std(vars_for_confidence)+np.mean(vars_for_confidence),2*np.std(vars_for_confidence)+np.mean(vars_for_confidence)#mean_confidence_interval(vars_for_confidence)
vars_within_conf = [(vars_conf[1]>i>vars_conf[0]) for i in sliding_vars]
vars_withinconf_fraction = np.array([sum(vars_within_conf[pt:]) / (len(sliding_vars) - pt) for pt in range(len(vars_for_confidence))])

T_for_confidence = [i  for i in sliding_T[int(len(sliding_T)/2):] if type(i) !=type(None)]
T_conf = -2*mean_confidence_interval(T_for_confidence)[0]+np.mean(T_for_confidence),2*mean_confidence_interval(T_for_confidence)[1]+np.mean(sliding_T)#mean_confidence_interval(T_for_confidence)
T_within_conf = [(T_conf[1]>i>T_conf[0]) if type(i)!=type(None) else False for i in sliding_T ]
T_withinconf_fraction = np.array([sum(T_within_conf[pt:]) / (len(sliding_T) - pt) for pt in range(len(T_for_confidence))])

means_for_confidence = sliding_means[int(len(sliding_means)/2):]
means_conf = -2*mean_confidence_interval(means_for_confidence)[0]+np.mean(means_for_confidence),2*mean_confidence_interval(means_for_confidence)[1]+np.mean(means_for_confidence)##mean_confidence_interval(means_for_confidence)
means_within_conf = [True if i>means_conf[0] and i<means_conf[1] else False for i in sliding_means ]
means_withinconf_fraction = np.array([sum(means_within_conf[pt:]) / (len(sliding_means) - pt) for pt in range(len(means_for_confidence))])




num_bins=50
slides_for_confidence = slides[second_half_timestepid:]
probabilities_for_confidence = [np.histogram(i,bins=num_bins)[0] for i in slides_for_confidence]

confidence = 0.95
mstat = np.where(means_withinconf_fraction > confidence)[0]
Tstat = np.where(T_withinconf_fraction > confidence)[0]
varstat = np.where(vars_withinconf_fraction > confidence)[0]



timestep_stationarity = max([mstat[0],Tstat[0],varstat[0]])*T
plt.plot(time,signal)
plt.axvspan(0.1, 0.1+T*dt, alpha=0.5, color='red')
plt.axvline(time[timestep_stationarity],color="k",linestyle="dashed")
plt.show()

