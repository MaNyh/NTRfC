import tempfile

import pyvista as pv

from ntrfc.filehandling.mesh import load_mesh
from ntrfc.turbo.cascade_case.casemeta.casemeta import CaseMeta
from ntrfc.turbo.cascade_case.utils.domain_utils import CascadeDomain2DParameters, Blade2D
from ntrfc.turbo.cascade_case.utils.probecontainer import ProbeContainer
from ntrfc.turbo.cascade_case.utils.sliceseries import SliceSeries


class GenericCascadeCase():
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

    def __init__(self, case_root_directory=None):
        super().__init__()
        if case_root_directory:
            self.case_meta = CaseMeta(case_root_directory)
        else:
            self.case_meta = CaseMeta(tempfile.mkdtemp())

        self.mesh_dict = {
            "inlet": pv.PolyData(),
            "outlet": pv.PolyData(),
            "blade": pv.PolyData(),
            "fluid": pv.UnstructuredGrid(),
            "yper_low": pv.PolyData(),
            "yper_high": pv.PolyData(),
            "zper_low": pv.PolyData(),
            "zper_high": pv.PolyData(),
        }

        self.sliceseries = SliceSeries()
        self.probes = ProbeContainer()
        self.domainparams = CascadeDomain2DParameters()
        self.blade = None
        self.active_blade_slice = pv.PolyData()

    def read_meshes(self, path, name):
        """
        Read data for any region from a file and store it in the mesh_dict.

        Args:
            path (str): Path to the file containing the geometry data.
        """

        self.mesh_dict[name] = load_mesh(path)

    def set_bladeslice_midz(self, z=None, alpha=None):
        if not z:
            bounds = self.mesh_dict["blade"].bounds
            z = bounds[4] + (bounds[5] - bounds[4]) / 2
        self.active_blade_slice = self.mesh_dict["blade"].slice(normal="z", origin=(0, 0, z))
        bladepoly = pv.PolyData(self.active_blade_slice.points)
        for arr in self.active_blade_slice.point_data.keys():
            bladepoly[arr] = self.active_blade_slice.point_data[arr]

        self.blade = Blade2D(bladepoly)
        self.blade.compute_all_frompoints(alpha)
