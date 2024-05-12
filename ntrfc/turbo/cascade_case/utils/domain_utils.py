import tempfile
from dataclasses import dataclass

import numpy as np
import pyvista as pv

from ntrfc.geometry.alphashape import auto_concaveHull, calc_concavehull
from ntrfc.geometry.line import polyline_from_points
from ntrfc.math.vectorcalc import vecAbs, vecAngle
from ntrfc.turbo.pointcloud_methods import is_counterclockwise, extractSidePolys, midline_from_sides
from ntrfc.turbo.profile_tele_extraction import extract_vk_hk


class Blade2D:
    def __init__(self, points: np.ndarray or pv.PolyData):
        self.origpoints_pv = None
        self.__sortedpoints_pv = None
        self.sortedpointsrolled_pv = None
        self.periodicsplinepoints = None
        self.skeletonline_pv = None
        self.ps_pv = None
        self.ss_pv = None
        self.hullalpha = None
        self.__ile_orig = None
        self.__ite_orig = None
        self.ile = None
        self.ite = None
        self.beta_le = None
        self.beta_te = None
        self.camber_phi = None
        self.camber_length = None

        if isinstance(points, pv.PolyData):
            self.origpoints_pv = points
            print("working with pv.PolyData, dataarrays can be used for postprocessing")
        elif isinstance(points, np.ndarray):
            self.origpoints_pv = pv.PolyData(points)
            print("working with np.ndarray pointcloud. No dataarrays cab be used for postprocessing")

    def compute_polygon(self, alpha=None):
        if alpha:
            xs, ys = calc_concavehull(self.origpoints_pv.points[:, 0], self.origpoints_pv.points[:, 1], alpha)
        else:
            xs, ys, alpha = auto_concaveHull(self.origpoints_pv.points[:, 0], self.origpoints_pv.points[:, 1])

        line = polyline_from_points(np.stack((xs, ys, np.zeros(len(ys)))).T)
        orientation = is_counterclockwise(xs, ys)

        # i need to find the indices of each xs,ys in self.origpoints_pv.points
        index_sort = []
        for xx, yy, __ in line.points:
            index_sort.append(np.where(np.all(np.array([xx, yy] == self.origpoints_pv.points[:, :2]), axis=1))[0][0])

        if not orientation:
            index_sort = index_sort[::-1]

        self.hullalpha = alpha
        self.__sortedpoints_pv = pv.PolyData()

        for i in index_sort:
            self.__sortedpoints_pv = self.__sortedpoints_pv.merge(self.origpoints_pv.extract_points([i]))

    def compute_lete(self):
        self.__ile_orig, self.__ite_orig = extract_vk_hk(self.__sortedpoints_pv)

    def compute_rolledblade(self):
        ids = np.roll(np.arange(self.__sortedpoints_pv.number_of_points), -self.__ile_orig)

        newSortedPoly = pv.PolyData()
        for i in ids:
            newSortedPoly = newSortedPoly.merge(self.__sortedpoints_pv.extract_points(i))

        self.ite = (self.__ile_orig - self.__ite_orig) % newSortedPoly.number_of_points
        self.ile = 0
        self.sortedpointsrolled_pv = newSortedPoly

    def extract_sides(self):
        self.ps_pv, self.ss_pv = extractSidePolys(self.ite, self.sortedpointsrolled_pv)

    def compute_skeleton(self):
        self.skeletonline_pv = midline_from_sides(self.ps_pv, self.ss_pv)

    def compute_stats(self):
        vk_tangent = self.skeletonline_pv.points[0] - self.skeletonline_pv.points[1]
        hk_tangent = self.skeletonline_pv.points[-2] - self.skeletonline_pv.points[-1]
        chord = -self.sortedpointsrolled_pv.points[self.ile] + self.sortedpointsrolled_pv.points[self.ite]
        self.beta_le = vecAngle(vk_tangent, -np.array([1, 0, 0])) / np.pi * 180
        self.beta_te = vecAngle(hk_tangent, np.array([1, 0, 0])) / np.pi * 180
        self.camber_phi = vecAngle(chord, np.array([1, 0, 0])) / np.pi * 180
        self.camber_length = vecAbs(self.skeletonline_pv.points[0] - self.skeletonline_pv.points[-1])

    def compute_all_frompoints(self, alpha=None):
        self.compute_polygon(alpha)
        self.compute_lete()
        self.compute_rolledblade()
        self.extract_sides()
        self.compute_skeleton()
        self.compute_stats()

    def plot(self, figurepath=tempfile.mkdtemp() + "/plot.png", window_size=[2400, 2400]):
        sfactor = window_size[0] / 200
        p = pv.Plotter(off_screen=True)
        p.add_mesh(self.ps_pv, color="red", label="pressure side", line_width=1)
        p.add_mesh(self.ss_pv, color="blue", label="suction side", line_width=1)
        p.add_mesh(self.skeletonline_pv, color="black", label="skeleton line", line_width=1)
        p.add_mesh(pv.PolyData(self.sortedpointsrolled_pv.points[self.ile]), color="green", label="ile", point_size=16)
        p.add_mesh(pv.PolyData(self.sortedpointsrolled_pv.points[self.ite]), color="yellow", label="ite", point_size=16)
        p.add_text(
            f"beta_le: {self.beta_le} \nbeta_te: {self.beta_te}\nphi_camber: {self.camber_phi}\nlength_camber: {self.camber_length}",
            font_size=int(sfactor))
        p.add_legend(size=(0.2, 0.2))
        p.view_xy()
        p.show(screenshot=figurepath, window_size=window_size)
        p.close()
        return figurepath


@dataclass
class CascadeDomain2DParameters:
    """
    A class representing the geometrical parameters of a simulation domain
    for a turbomachinery linear cascade simulation.

    Attributes:
        blade: Blade2D: A Blade2D object representing the blade.
        xinlet (float): The x coordinate of the inlet.
        xoutlet (float): The x coordinate of the outlet.
        pitch (float): The pitch of the blades.
        blade_yshift (float): The y shift of the blades.
        profile_points (pv.PolyData): A pv.PolyData object representing the profile points.
    """
    xinlet: float = None
    xoutlet: float = None
    pitch: float = None
    blade_yshift: float = None


@dataclass
class CascadeWindTunnelDomain2DParameters:
    """
    A class representing the domain of a turbomachinery linear cascade simulation.

    Attributes:
        casemeta (CaseMeta): A CaseMeta object representing the case metadata.
        blade (Blade2D): A Blade2D object representing the blade.
        xinlet (float): The x coordinate of the inlet.
        xoutlet (float): The x coordinate of the outlet.
        pitch (float): The pitch of the blades.
        blade_yshift (float): The y shift of the blades.
        profile_points (pv.PolyData): A pv.PolyData object representing the profile points.
    """
    gamma: float = None
    gittervor: float = None
    pitch: float = None
    gitternach: float = None
    zulauf: float = None
    tailbeta: float = None
    nblades: int = None
