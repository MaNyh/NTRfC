import os
from pathlib import Path

import numpy as np
from fluidfoam.readpostpro import readprobes

from ntrfc.cascade_case.solution.generic import GenericCascadeCase


class openfoam_case(GenericCascadeCase):
    def __init__(self):
        super().__init__()
        self.solver = "openfoam"

    def read_openfoam_probes(self, path_to_probes):
        """
        adds probes from an openfoam case to an ntrfc-case
        :param path_to_probes: points on the directory
        """
        openfoam_probesets_starttimes = os.listdir(path_to_probes)
        openfoam_probesets_starttimes.sort()
        postprocessing_dir = Path(path_to_probes).parent.parent
        basename = Path(path_to_probes).stem

        probe_time = {}
        probe_vals = {}
        timestamps = [float(i) for i in openfoam_probesets_starttimes]

        vals = os.listdir(os.path.join(path_to_probes, openfoam_probesets_starttimes[0]))
        for val in vals:
            _, times, values = readprobes(f"{postprocessing_dir}", basename, time_name="mergeTime", name=val)
            values[values == -1e300] = np.nan
            probe_time[val] = times
            probe_vals[val] = values

        for valname in probe_time.keys():
            self.probes.add_probeset(valname, basename, probe_time[valname], probe_vals[valname], timestamps)

    def read_openfoam_sliceseries(self, path_to_slices, filename):
        """
        adds slices to
        """
        timedirs = os.listdir(path_to_slices)
        timedirs.sort()

        groupname = os.path.basename(path_to_slices)
        paths = []
        timestamps = []
        for td in timedirs:
            files = os.listdir(os.path.join(path_to_slices, td))
            if filename in files:
                slicepath = os.path.join(path_to_slices, td, filename)
                print(f"adding {slicepath} to the slice-list")
                paths.append(slicepath)
                timestamps.append(td)
        print(f"creating sliceseries with {len(timestamps)} slices")
        self.sliceseries.add_sliceset(paths, groupname, timestamps)
