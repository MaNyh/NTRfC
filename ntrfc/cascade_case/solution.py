import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from ntrfc.cascade_case.casemeta.casemeta import CaseMeta
from ntrfc.cascade_case.utils.domain_utils import DomainParameters
from ntrfc.cascade_case.utils.probecontainer import ProbeContainer
from ntrfc.cascade_case.utils.sliceseries import SliceSeries
from ntrfc.filehandling.mesh import load_mesh
from ntrfc.geometry.plane import massflowave_plane
from ntrfc.math.vectorcalc import vecAngle, vecAbs
from ntrfc.turbo.bladeloading import calc_inflow_cp
from ntrfc.turbo.integrals import avdr


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
        self.domainparams = DomainParameters()
        self.active_blade_slice = pv.PolyData()

    def read_meshes(self, path, name):
        """
        Read data for any region from a file and store it in the mesh_dict.

        Args:
            path (str): Path to the file containing the geometry data.
        """

        self.mesh_dict[name] = load_mesh(path)

    def set_active_blade_slice(self, z=None):
        if not z:
            bounds = self.mesh_dict["blade"].bounds
            z = bounds[4] + (bounds[5] - bounds[4]) / 2
        self.active_blade_slice = self.mesh_dict["blade"].slice(normal="z", origin=(0, 0, z))

    def compute_domainparams_from(self, alpha):
        self.domainparams.generate_params_by_pointcloud(self.active_blade_slice, alpha)

    def compute_avdr_inout_massave(self):

        self.mesh_dict["inlet"]["u"] = self.mesh_dict["inlet"]["U"][::, 0]
        self.mesh_dict["inlet"]["v"] = self.mesh_dict["inlet"]["U"][::, 1]
        self.mesh_dict["inlet"]["w"] = self.mesh_dict["inlet"]["U"][::, 2]

        self.mesh_dict["outlet"]["u"] = self.mesh_dict["outlet"]["U"][::, 0]
        self.mesh_dict["outlet"]["v"] = self.mesh_dict["outlet"]["U"][::, 1]
        self.mesh_dict["outlet"]["w"] = self.mesh_dict["outlet"]["U"][::, 2]
        rho_1 = massflowave_plane(self.mesh_dict["inlet"], "rho")
        mag_u_1 = vecAbs(
            np.array([massflowave_plane(self.mesh_dict["inlet"], "u"), massflowave_plane(self.mesh_dict["inlet"], "v"),
                      massflowave_plane(self.mesh_dict["inlet"], "w")]))
        U_1 = np.stack(
            [massflowave_plane(self.mesh_dict["inlet"], "u"), massflowave_plane(self.mesh_dict["inlet"], "v"),
             massflowave_plane(self.mesh_dict["inlet"], "w")])
        beta_1 = vecAngle(U_1, np.array([1, 0, 0]))
        rho_2 = massflowave_plane(self.mesh_dict["outlet"], "rho")
        U_2 = np.stack(
            [massflowave_plane(self.mesh_dict["outlet"], "u"), massflowave_plane(self.mesh_dict["outlet"], "v"),
             massflowave_plane(self.mesh_dict["outlet"], "w")])
        mag_u_2 = vecAbs(np.array(
            [massflowave_plane(self.mesh_dict["outlet"], "u"), massflowave_plane(self.mesh_dict["outlet"], "v"),
             massflowave_plane(self.mesh_dict["outlet"], "w")]))
        beta_2 = vecAngle(U_2, np.array([1, 0, 0]))
        self.avdr = avdr(rho_1, mag_u_1, beta_1, rho_2, mag_u_2, beta_2)
        return self.avdr

    def blade_loading(self, figpath=tempfile.mkdtemp() + "/blade_loading.png"):
        sspoints = self.domainparams.sspoly.points
        pspoints = self.domainparams.pspoly.points

        inlet = self.mesh_dict["inlet"]
        inlet["u"] = inlet["U"][::, 0]
        inlet["v"] = inlet["U"][::, 1]
        inlet["w"] = inlet["U"][::, 2]
        p1 = massflowave_plane(inlet, valname="p", rhoname="rho", velocityname="U")
        rho = massflowave_plane(inlet, valname="rho", rhoname="rho", velocityname="U")
        u = massflowave_plane(inlet, valname="u", rhoname="rho", velocityname="U")
        v = massflowave_plane(inlet, valname="v", rhoname="rho", velocityname="U")
        w = massflowave_plane(inlet, valname="w", rhoname="rho", velocityname="U")
        U = vecAbs([u, v, w])
        pt1 = p1 + 1 / 2 * rho * U ** 2

        bladepoly = self.domainparams.profile_points

        ssmeshpointids = [bladepoly.find_closest_point(pt) for pt in sspoints]
        psmeshpointids = [bladepoly.find_closest_point(pt) for pt in pspoints]

        ssmeshpoints = bladepoly.extract_points(ssmeshpointids)
        psmeshpoints = bladepoly.extract_points(psmeshpointids)

        ind_le = self.domainparams.leading_edge_index
        ind_te = self.domainparams.trailing_edge_index
        ssmeshpoints.points -= bladepoly.points[ind_le]
        psmeshpoints.points -= bladepoly.points[ind_le]
        bladepoly.points -= bladepoly.points[ind_le]
        camber_length = vecAbs(bladepoly.points[ind_le] - bladepoly.points[ind_te])

        camber_angle = self.domainparams.stagger_angle

        ssmeshpoints.rotate_z(camber_angle, inplace=True)
        psmeshpoints.rotate_z(camber_angle, inplace=True)

        ps_xc = np.zeros(psmeshpoints.number_of_points)
        ps_cp = np.zeros(psmeshpoints.number_of_points)

        for idx, pts1 in enumerate(psmeshpoints.points):
            ps_xc[idx] = pts1[0] / camber_length
            ps_cp[idx] = calc_inflow_cp(psmeshpoints.point_data["p"][idx], pt1, p1)

        ss_xc = np.zeros(ssmeshpoints.number_of_points)
        ss_cp = np.zeros(ssmeshpoints.number_of_points)

        for idx, pts1 in enumerate(ssmeshpoints.points):
            ss_xc[idx] = pts1[0] / camber_length
            ss_cp[idx] = calc_inflow_cp(ssmeshpoints.point_data["p"][idx], pt1, p1)

        plt.figure()
        plt.title("blade loading")
        plt.scatter(ss_xc, ss_cp, label="suction side")
        plt.scatter(ps_xc, ps_cp, label="pressure side")
        plt.xlabel("$x/c_{ax}$")
        plt.ylabel("$c_{p}$")
        plt.grid()
        plt.legend()
        plt.savefig(figpath)

        return 0
