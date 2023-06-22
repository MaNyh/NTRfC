from dataclasses import dataclass

import pyvista as pv

from ntrfc.turbo.pointcloud_methods import extract_geo_paras


@dataclass
class DomainParameters:
    """
    A class representing the geometrical parameters of a simulation domain
    for a turbomachinery linear cascade simulation.

    Attributes:
        stagger_angle (float): The stagger angle of the blades.
        sspoly (pv.PolyData): A pv.PolyData object representing the suction side poly.
        pspoly (pv.PolyData): A pv.PolyData object representing the pressure side poly.
        xinlet (float): The x coordinate of the inlet.
        xoutlet (float): The x coordinate of the outlet.
        pitch (float): The pitch of the blades.
        blade_yshift (float): The y shift of the blades.
        beta_in (float): The metal angle of the profile at the inlet.
        beta_out (float): The metal angle of the profile at the outlet.
        alpha (float): The alpha shape of the blade.
        profile_points (pv.PolyData): A pv.PolyData object representing the profile points.
        leading_edge_index (int): The index of the leading edge point in the profile_points array.
        trailing_edge_index (int): The index of the trailing edge point in the profile_points array.
    """
    stagger_angle: float = None
    sspoly: pv.PolyData = None
    pspoly: pv.PolyData = None
    midspoly: pv.PolyData = None
    xinlet: float = None
    xoutlet: float = None
    pitch: float = None
    blade_yshift: float = None
    beta_in: float = None
    beta_out: float = None
    alpha: float = None
    profile_points: pv.PolyData = None
    leading_edge_index: int = None
    trailing_edge_index: int = None

    def plot_domainparas(self):
        """
        Plot the domain parameters using PyVista.


        Returns:
            pv.Plotter: The PyVista plotter object used for plotting.
        """
        plotter = pv.Plotter()
        plotter.window_size = 6400, 6400
        # Plot the suction side and pressure side polys
        plotter.add_mesh(self.sspoly, color='b', show_edges=True, label="suction-side")
        plotter.add_mesh(self.pspoly, color='r', show_edges=True, label="pressure-side")
        plotter.add_mesh(self.midspoly, color='k', label="camber")

        # Plot the profile points
        plotter.add_points(self.profile_points, point_size=1, color='w', opacity=0.8)
        plotter.add_legend()
        plotter.add_axes()
        plotter.show(cpos=(0, 0, 1))

    def generate_params_by_pointcloud(self, points, alpha=None):
        sortedPoly, psPoly, ssPoly, ind_vk, ind_hk, midsPoly, beta_leading, beta_trailing, camber_angle, alpha = extract_geo_paras(
            points, alpha)

        self.profile_points = sortedPoly
        self.sspoly = ssPoly
        self.pspoly = psPoly
        self.midspoly = midsPoly
        self.leading_edge_index = ind_vk
        self.trailing_edge_index = ind_hk
        self.beta_in = beta_leading
        self.beta_out = beta_trailing
        self.stagger_angle = camber_angle
        self.alpha = alpha
