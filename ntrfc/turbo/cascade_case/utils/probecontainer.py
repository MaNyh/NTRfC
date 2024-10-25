import tempfile
from dataclasses import dataclass, field
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Probe:
    position: np.array
    values: dict
    groupname: str
    timesteps: np.array


@dataclass
class ProbeContainer:
    # i need a default value for probes with an empty list
    probes: list = field(default_factory=lambda: [])

    # add function that appends a new probe to the list of probes

    def add_probe(self, position, values, groupname, timesteps):
        self.probes.append(Probe(position, values, groupname, timesteps))

    def filter_probes(self, by_groupname=None, by_position=None):
        filtered_probes = self.probes

        if by_groupname:
            filtered_probes = [probe for probe in filtered_probes if probe.groupname == by_groupname]

        if by_position is not None:
            filtered_probes = [probe for probe in filtered_probes if np.array_equal(probe.position, by_position)]

        return filtered_probes

    def plot_probes(self, groupname=None, positions=None, value=None, path=None):
        assert value, "a value must be specified"
        if path is None:
            tmpdir = tempfile.mkdtemp()
            path = tmpdir + "/plot.png"
        plotprobelist = self.filter_probes(groupname, positions)
        plt.figure(figsize=(32, 24), dpi=200)
        for probe in plotprobelist:
            if probe.values[value][0].shape == (1,):
                plt.plot(probe.timesteps, probe.values[value].T[0], label=f"{probe.position}")
            elif probe.values[value][0].shape == (3,):
                color = tuple(np.random.rand(3))
                plt.plot(probe.timesteps, probe.values[value][:, 0], label=f"{probe.position}", color=color)
                plt.plot(probe.timesteps, probe.values[value][:, 1], color=color)
                plt.plot(probe.timesteps, probe.values[value][:, 2], color=color)
        plt.legend()
        plt.savefig(path)
