import numpy as np
import pyvista as pv

from ntrfc.cascade_case.solution.case_modules.postprocessing import PostProcessing
from ntrfc.cascade_case.solution.case_modules.probecontainer import ProbeContainer
from ntrfc.cascade_case.solution.case_modules.sliceseries import SliceSeries
from ntrfc.cascade_case.domain.domain import DomainParameters
from ntrfc.filehandling.mesh import load_mesh


class GenericCascadeCase(PostProcessing ):
    """A container for data related to a cascade case, including geometry data and fluid flow data.

    This class provides functionality for reading in data from file and storing it in instance variables, as well as
    postprocessing, defining a probe proberegistry, and defining a sliceseriesregistry using inherited classes.

    Attributes:
        solver (object): An object representing the solver used to generate the data for this case.
        inlet (pv.PolyData): Geometry data for the inlet region of the case.
        outlet (pv.PolyData): Geometry data for the outlet region of the case.
        blade (pv.PolyData): Geometry data for the blade in the case.
        fluid (pv.UnstructuredGrid): Fluid flow data for the case.
        yper_low (pv.PolyData): Geometry data for the lower y-perpendicular plane of the case.
        yper_high (pv.PolyData): Geometry data for the upper y-perpendicular plane of the case.
        zper_low (pv.PolyData): Geometry data for the lower z-perpendicular plane of the case.
        zper_high (pv.PolyData): Geometry data for the upper z-perpendicular plane of the case.
        probes (ProbeContainer): A registry of probes defined for this case.
        sliceseries (SliceSeries): A registry of slices defined for this case.
    """
    def __init__(self):
        super().__init__()
        self.solver = None

        self.inlet = pv.PolyData()
        self.outlet = pv.PolyData()
        self.blade = pv.PolyData()
        self.fluid = pv.UnstructuredGrid()
        self.yper_low = pv.PolyData()
        self.yper_high = pv.PolyData()
        self.zper_low = pv.PolyData()
        self.zper_high = pv.PolyData()

        self.probes = ProbeContainer()
        self.sliceseries = SliceSeries()

        self.domainparams = DomainParameters()

        self.active_blade_slice = pv.PolyData()

    def read_inlet(self, path):
        """
        Read geometry data for the inlet region from a file and store it in the 'inlet' attribute.
        Ensure that all surface normals are aligned with the x-axis.

        Args:
            path (str): Path to the file containing the inlet geometry data.
        """
        inlet = load_mesh(path)

        normals = inlet.compute_normals()["Normals"]
        if not all(np.isclose(normals[:, 0], -1)):
            raise ValueError("Not all surface normals are aligned with the x-axis.")

        self.inlet = inlet

    def read_outlet(self, path):
        """
        Read geometry data for the outlet region from a file and store it in the 'outlet' attribute.
        Ensure that all surface normals are aligned with the x-axis.

        Args:
            path (str): Path to the file containing the outlet geometry data.
        """
        outlet = load_mesh(path)

        normals = outlet.compute_normals()["Normals"]
        if not all(np.isclose(normals[:, 0], 1)):
            raise ValueError("Not all surface normals are aligned with the x-axis.")

        self.outlet = outlet

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

    def read_fluid(self,path):
        self.fluid = load_mesh(path)

    def set_active_blade_slice(self,z=None):
        if not z:
            bounds = self.blade.bounds
            z = bounds[4]+(bounds[5]-bounds[4])/2
        self.active_blade_slice = self.blade.slice(normal="z",origin=(0,0,z))

    def compute_domainparams_from(self,alpha):
        self.domainparams.generate_params_by_pointcloud(self.active_blade_slice,alpha)
