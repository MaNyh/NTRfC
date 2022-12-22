from dataclasses import dataclass, field

import numpy as np
from matplotlib import pyplot as plt


@dataclass
class ProbeContainer:
    probesets: list = field(default_factory=list)

    def add_probeset(self, variablename, groupname, time, values, timestamps=[]):
        new_probeset = probeset(variablename, groupname, time, values, timestamps)
        self.probesets.append(new_probeset)

    def plot_all(self, id="all", save=None):
        """
        :param id: str (all), int (id)
        """
        plt.figure(figsize=(32, 24), dpi=200)
        for probeset in self.probesets:
            nprobes = probeset.arrays.shape[1]
            if id == "all":
                for pidx in range(nprobes):
                    probedata = probeset.arrays[:, pidx]
                    ndim = probedata.shape
                    if len(ndim) > 1:

                        for dim in range(ndim[1]):
                            probedatacolumn = probedata[::, dim]
                            plt.plot(probeset.time, probedatacolumn, label=f"Probe ID {pidx}")

            elif type(id) == int and id > 0:
                probedata = probeset.arrays[:, id]
                ndim = probedata.shape
                if len(ndim) > 1:

                    for dim in range(ndim[1]):
                        probedatacolumn = probedata[::, dim]
                        plt.plot(probeset.time, probedatacolumn, label=f"Probe ID {id}")

            for ts in probeset.timestamps:
                plt.vlines(ts, ymin=np.nanmin(probeset.arrays), ymax=np.nanmax(probeset.arrays),
                           linestyles="dotted", colors="red")
            plt.title(f"{probeset.variablename}~{probeset.groupname}")
            plt.xlabel("time [s]")
            plt.ylabel(f"{probeset.variablename}")
            plt.grid()
            plt.legend()
            plt.show()
            if save:
                plt.savefig(f"{save}/{probeset.variablename}~{probeset.groupname}.svg")
            plt.close()

    def plot_val(self, val, save=None):
        plt.figure(figsize=(32, 24), dpi=200)
        for probeset in [i for i in self.probesets if i.variablename == val]:
            nprobes = probeset.arrays.shape[1]
            for pidx in range(nprobes):
                probedata = probeset.arrays[:, pidx]
                ndim = probedata.shape
                if len(ndim) > 1:

                    for dim in range(ndim[1]):
                        probedatacolumn = probedata[::, dim]
                        plt.plot(probeset.time, probedatacolumn)

                for ts in probeset.timestamps:
                    plt.vlines(ts, ymin=np.nanmin(probeset.arrays), ymax=np.nanmax(probeset.arrays),
                               linestyles="dotted", colors="red")
            plt.title(f"{probeset.variablename}~{probeset.groupname}")
            plt.xlabel("time [s]")
            plt.ylabel(f"{probeset.variablename}")
            plt.grid()
            plt.legend()
            plt.show()
            if save:
                plt.savefig(f"{save}/{probeset.variablename}~{probeset.groupname}.svg")
            plt.close()


@dataclass
class probeset:
    '''Object for tracking probes a set.'''
    variablename: str
    groupname: str
    time: np.array
    arrays: np.array
    timestamps: list
