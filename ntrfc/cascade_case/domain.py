import os
import tempfile
from dataclasses import dataclass

import numpy as np
import pyvista as pv

from ntrfc.cascade_case.casemeta.casemeta import CaseMeta
from ntrfc.cascade_case.utils.domain_utils import DomainParameters
from ntrfc.math.vectorcalc import vecAbs
from ntrfc.turbo.pointcloud_methods import calcMidPassageStreamLine


@dataclass
class CascadeDomain2D:
    casemeta: CaseMeta = CaseMeta(tempfile.mkdtemp())
    pressureside: pv.PolyData = None
    suctionside: pv.PolyData = None
    profilepoints: pv.PolyData = None
    le_index: int = None
    te_index: int = None
    beta_leading: float = None
    beta_trailing: float = None
    yperiodic_low: pv.PolyData = None
    yperiodic_high: pv.PolyData = None
    inlet: pv.PolyData = None
    outlet: pv.PolyData = None
    pitch: float = None
    chordlength: float = None

    def generate_from_cascade_parameters(self, domainparams: DomainParameters):
        # Use params attributes to generate attributes of CascadeDomain2D
        self.profilepoints = domainparams.profile_points
        self.le_index = domainparams.leading_edge_index
        self.te_index = domainparams.trailing_edge_index
        x_mids = domainparams.midspoly.points[::, 0]
        y_mids = domainparams.midspoly.points[::, 1]
        self.beta_leading = domainparams.beta_in
        self.beta_trailing = domainparams.beta_out
        x_inlet = domainparams.xinlet
        x_outlet = domainparams.xoutlet
        self.pitch = domainparams.pitch
        blade_shift = domainparams.blade_yshift

        x_mpsl, y_mpsl = calcMidPassageStreamLine(x_mids, y_mids, self.beta_leading, self.beta_trailing,
                                                  x_inlet, x_outlet, self.pitch)

        y_upper = np.array(y_mpsl) + blade_shift
        per_y_upper = pv.lines_from_points(np.stack((np.array(x_mpsl),
                                                     np.array(y_upper),
                                                     np.zeros(len(x_mpsl)))).T)
        y_lower = y_upper - self.pitch
        per_y_lower = pv.lines_from_points(np.stack((np.array(x_mpsl),
                                                     np.array(y_lower),
                                                     np.zeros(len(x_mpsl)))).T)

        inlet_pts = np.array([per_y_lower.points[per_y_lower.points[::, 0].argmin()],
                              per_y_upper.points[per_y_upper.points[::, 0].argmin()]])

        inletPoly = pv.Line(*inlet_pts)
        outlet_pts = np.array([per_y_lower.points[per_y_lower.points[::, 0].argmax()],
                               per_y_upper.points[per_y_upper.points[::, 0].argmax()]])

        outletPoly = pv.Line(*outlet_pts)

        self.pressureside = domainparams.pspoly
        self.suctionside = domainparams.sspoly
        self.yperiodic_low = per_y_lower
        self.yperiodic_high = per_y_upper
        self.inlet = inletPoly
        self.outlet = outletPoly
        self.chordlength = vecAbs(self.profilepoints.points[self.te_index] - self.profilepoints.points[self.le_index])

    def plot_domain(self):
        """
        Plot the domain parameters using PyVista.


        Returns:
            pv.Plotter: The PyVista plotter object used for plotting.
        """
        if os.getenv('DISPLAY') is None:
            pv.start_xvfb()  # Start X virtual framebuffer (Xvfb)
        plotter = pv.Plotter(off_screen=True)
        path =os.path.join(self.casemeta.case_root_directory, "domain.png")

        plotter.window_size = 2400, 2400
        # Plot the suction side and pressure side polys
        plotter.add_mesh(self.suctionside, color='b', show_edges=True)
        plotter.add_mesh(self.pressureside, color='r', show_edges=True)
        plotter.add_mesh(self.yperiodic_low)
        plotter.add_mesh(self.yperiodic_high)
        plotter.add_mesh(self.inlet)
        plotter.add_mesh(self.outlet)

        plotter.add_axes()
        plotter.view_xy()
        plotter.screenshot(path)
        return path
