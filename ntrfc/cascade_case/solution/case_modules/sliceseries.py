import os
from dataclasses import dataclass, field

import pyvista as pv
from tqdm import tqdm

from ntrfc.filehandling.mesh import load_mesh


@dataclass
class SliceSeries:
    """
    object for storing pathes to slices
    """
    slicesets: list = field(default_factory=list)

    def add_sliceset(self, slices, groupname, timestamps):
        self.slicesets.append(sliceset(slices, groupname, timestamps))

    def create_animation(self, slicegroup, variable, path, name):
        slicegroups = [self.slicesets[i].groupname for i in range(len(self.slicesets))]
        use_idx = slicegroups.index(slicegroup)
        use_slices = self.slicesets[use_idx].slices
        use_timestamps = self.slicesets[use_idx].timestamps

        pv.set_plot_theme("document")
        fontsize = 36
        labelsize = 36

        gifpath = os.path.join(path, name)

        first_slice = load_mesh(use_slices[0])

        p = pv.Plotter(notebook=False, off_screen=True)
        p.window_size = [3840, 2160]
        p.open_gif(gifpath)
        p.add_mesh(first_slice, scalars=variable)
        p.add_text(f"framenumber: 0 - time {use_timestamps[0]}", font_size=labelsize, position='upper_left')
        p.camera.position = (0, 0, 1)
        p.camera.roll = 0
        p.camera.zoom = 1

        for idx, (path, time) in tqdm(enumerate(zip(use_slices[1:], use_timestamps[1:]))):
            sliceframe = load_mesh(path)
            p.clear()
            p.add_mesh(sliceframe, scalars=variable)
            p.add_text(f"framenumber: {idx} - time {time}", font_size=fontsize, position='upper_left')
            p.scalar_bar.SetTitleRatio(2)
            p.write_frame()

        # Closes and finalizes movie
        p.close()


@dataclass
class sliceset:
    '''Object for tracking probes a set.'''
    slices: list
    groupname: str
    timestamps: list
