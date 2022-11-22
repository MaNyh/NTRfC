import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import os
from tqdm import tqdm

from ntrfc.filehandling.mesh import load_mesh
from ntrfc.geometry.plane import massflowave_plane
from ntrfc.math.vectorcalc import vecAngle
from ntrfc.turbo.integrals import avdr



@dataclass
class proberegistry:
    probesets: list = field(default_factory=list)

    def add_probeset(self,variablename,groupname,time,values,timestamps=[]):
        new_probeset = probeset(variablename,groupname, time, values, timestamps)
        self.probesets.append(new_probeset)

    def plot_all(self,id = "all"):
        """
        :param id: str (all), int (id)
        """
        plt.figure(figsize=(32, 24),dpi=200)
        for probeset in self.probesets:
            nprobes = probeset.arrays.shape[1]
            if id== "all":
                for pidx in range(nprobes):
                    probedata = probeset.arrays[:,pidx]
                    ndim = probedata.shape
                    if len(ndim)>1:

                        for dim in range(ndim[1]):
                            probedatacolumn = probedata[::,dim]
                            plt.plot(probeset.time,probedatacolumn,label=f"Probe ID {pidx}")

            elif type(id)==int and id>0:
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
            plt.close()

    def plot_val(self,val):
        plt.figure(figsize=(32, 24),dpi=200)
        for probeset in [i for i in self.probesets if i.variablename==val]:
            nprobes = probeset.arrays.shape[1]
            for pidx in range(nprobes):
                probedata = probeset.arrays[:,pidx]
                ndim = probedata.shape
                if len(ndim)>1:

                    for dim in range(ndim[1]):
                        probedatacolumn = probedata[::,dim]
                        plt.plot(probeset.time,probedatacolumn)

                for ts in probeset.timestamps:
                    plt.vlines(ts,ymin=np.nanmin(probeset.arrays),ymax=np.nanmax(probeset.arrays),
                               linestyles = "dotted" , colors = "red")
            plt.title(f"{probeset.variablename}~{probeset.groupname}")
            plt.xlabel("time [s]")
            plt.ylabel(f"{probeset.variablename}")
            plt.grid()
            plt.legend()
            plt.show()
            plt.close()

@dataclass
class probeset:
    '''Object for tracking probes a set.'''
    variablename: str
    groupname: str
    time: np.array
    arrays: np.array
    timestamps: list

@dataclass
class sliceregistry:
    """
    object for storing pathes to slices
    """
    slicesets: list = field(default_factory=list)

    def add_sliceset(self,slices,groupname,timestamps):
        self.slicesets.append(sliceset(slices,groupname,timestamps))

    def create_animation(self,slicegroup,variable,path,name):
        slicegroups = [self.slicesets[i].groupname for i in range(len(self.slicesets))]
        use_idx = slicegroups.index(slicegroup)
        use_slices = self.slicesets[use_idx].slices
        use_timestamps =  self.slicesets[use_idx].timestamps

        pv.set_plot_theme("document")
        fontsize = 36
        labelsize = 36
        pv.global_theme.font.label_size = labelsize

        gifpath = os.path.join(path,name)

        first_slice = load_mesh(use_slices[0])

        p = pv.Plotter(notebook=False, off_screen=True)
        p.window_size = [3840, 2160]
        p.open_gif(gifpath)
        p.add_mesh(first_slice, scalars=variable)
        p.add_text(f"framenumber: 0 - time {use_timestamps[0]}", font_size=fontsize)
        p.camera.position = (0, 0, 1)
        p.camera.roll = 0
        p.camera.zoom = 1

        for idx, (path,time) in tqdm(enumerate(zip(use_slices[1:],use_timestamps[1:]))):

            sliceframe = load_mesh(path)
            p.clear()
            p.add_mesh(sliceframe, scalars=variable)
            p.add_text(f"framenumber: {idx} - time {time}", font_size=fontsize)
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

class solution:

    def __init__(self):
        self.solver = None

        self.inlet = pv.PolyData()
        self.outlet = pv.PolyData()
        self.blade = pv.PolyData()
        self.fluid = pv.UnstructuredGrid()
        self.yper_low = pv.PolyData()
        self.yper_high = pv.PolyData()
        self.zper_low = pv.PolyData()
        self.zper_high = pv.PolyData()

        self.probes = proberegistry()
        self.sliceseries = sliceregistry()



    def read_fluid(self, path):
        self.fluid = load_mesh(path)

    def read_inlet(self, path):
        self.inlet = load_mesh(path)

    def read_outlet(self, path):
        self.outlet = load_mesh(path)

    def read_blade(self, path):
        self.blade = load_mesh(path)

    def read_yper_low(self, path):
        self.yper_low = load_mesh(path)

    def read_yper_high(self, path):
        self.yper_high = load_mesh(path)

    def read_zper_low(self, path):
        self.zper_low = load_mesh(path)

    def read_zper_high(self, path):
        self.zper_high = load_mesh(path)

    def compute_avdr(self):
        rho_1 = massflowave_plane(self.inlet, "rho")
        mag_u_1 = massflowave_plane(self.inlet, "U")
        U_1 = np.stack([massflowave_plane(self.inlet, "u"), massflowave_plane(self.inlet, "v"),
                        massflowave_plane(self.inlet, "w")])
        beta_1 = vecAngle(U_1, np.array([1, 0, 0]))
        rho_2 = massflowave_plane(self.outlet, "rho")
        U_2 = np.stack([massflowave_plane(self.inlet, "u"), massflowave_plane(self.inlet, "v"),
                        massflowave_plane(self.inlet, "w")])
        mag_u_2 = massflowave_plane(self.outlet, "U")
        beta_2 = vecAngle(U_2, np.array([1, 0, 0]))
        self.avdr = avdr(rho_1, mag_u_1, beta_1, rho_2, mag_u_2, beta_2)


