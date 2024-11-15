import tempfile
from dataclasses import dataclass
import pyvista as pv
import numpy as np

from ntrfc.filehandling.mesh import load_mesh
from ntrfc.turbo.cascade_case.casemeta.casemeta import CaseMeta
from ntrfc.turbo.cascade_case.utils.domain_utils import CascadeDomain2DParameters, Blade2D
from ntrfc.turbo.cascade_case.utils.probecontainer import ProbeContainer
from ntrfc.turbo.cascade_case.utils.sliceseries import SliceSeries


@dataclass
class CaseStatistics:
    """
    A container for statistics data for a case.
    """

    ps_x: np.ndarray = None
    ps_pressure: np.ndarray = None
    ps_cp: np.ndarray = None
    ps_deltax: np.ndarray = None
    ps_deltay: np.ndarray = None
    ps_deltaz: np.ndarray = None
    ps_mais: np.ndarray = None

    ss_x: np.ndarray = None
    ss_pressure: np.ndarray = None
    ss_cp: np.ndarray = None
    ss_deltax: np.ndarray = None
    ss_deltay: np.ndarray = None
    ss_deltaz: np.ndarray = None
    ss_mais: np.ndarray = None

    avdr: float = None

    aspect_ratios: np.ndarray = None
    skewness: np.ndarray = None
    expansion_factors: np.ndarray = None

    turbulentintensity_tux: np.ndarray = None
    turbulentintensity_x: np.ndarray = None


class GenericCascadeCase():
    """A container for data related to a cascade case, including geometry data and fluid flow data.

    This class provides functionality for reading in data from file and storing it in instance variables, as well as
    postprocessing, defining a probe proberegistry, and defining a sliceseriesregistry using inherited classes.

    Attributes:
    case_meta (CaseMeta): A CaseMeta object containing metadata for the case.
    mesh_dict (dict): A dictionary containing the geometry data for the case.
    sliceseries (SliceSeries): A SliceSeries object containing the slice series for the case.
    probes (ProbeContainer): A ProbeContainer object containing the probe registry for the case.
    statistics (CaseStatistics): A CaseStatistics object containing the statistics for the case.
    domainparams (CascadeDomain2DParameters): A CascadeDomain2DParameters object containing the domain parameters for the case.
    blade (Blade2D): A Blade2D object containing the blade geometry for the case.

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
        self.statistics = CaseStatistics()
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
