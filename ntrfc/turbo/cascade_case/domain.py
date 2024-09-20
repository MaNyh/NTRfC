import os
import tempfile
from dataclasses import dataclass

import numpy as np
import pyvista as pv

from ntrfc.geometry.line import lines_from_points
from ntrfc.math.vectorcalc import line_intersection, vecAbs
from ntrfc.turbo.cascade_case.casemeta.casemeta import CaseMeta
from ntrfc.turbo.cascade_case.utils.domain_utils import Blade2D
from ntrfc.turbo.cascade_case.utils.domain_utils import CascadeDomain2DParameters, CascadeWindTunnelDomain2DParameters
from ntrfc.turbo.pointcloud_methods import calcMidPassageStreamLine


@dataclass
class CascadeDomain2D:
    casemeta: CaseMeta = CaseMeta(tempfile.mkdtemp())
    blade: Blade2D = None
    domainparams: CascadeDomain2DParameters = None
    yperiodic_low: pv.PolyData = None
    yperiodic_high: pv.PolyData = None
    inlet: pv.PolyData = None
    outlet: pv.PolyData = None

    # pitch: float = None
    # chordlength: float = None

    def generate_from_cascade_parameters(self, domainparams: CascadeDomain2DParameters, blade: Blade2D):
        # Use params attributes to generate attributes of CascadeDomain2D
        self.blade = blade
        x_mids = blade.skeletonline_pv.points[::, 0]
        y_mids = blade.skeletonline_pv.points[::, 1]
        beta_le = blade.beta_le
        beta_te = blade.beta_te
        x_inlet = domainparams.xinlet
        x_outlet = domainparams.xoutlet
        blade_shift = domainparams.blade_yshift

        x_mpsl, y_mpsl = calcMidPassageStreamLine(x_mids, y_mids, beta_le, beta_te,
                                                  x_inlet, x_outlet, domainparams.pitch)

        y_upper = np.array(y_mpsl) + blade_shift
        per_y_upper = pv.lines_from_points(np.stack((np.array(x_mpsl),
                                                     np.array(y_upper),
                                                     np.zeros(len(x_mpsl)))).T)
        y_lower = y_upper - domainparams.pitch
        per_y_lower = pv.lines_from_points(np.stack((np.array(x_mpsl),
                                                     np.array(y_lower),
                                                     np.zeros(len(x_mpsl)))).T)

        inlet_pts = np.array([per_y_lower.points[per_y_lower.points[::, 0].argmin()],
                              per_y_upper.points[per_y_upper.points[::, 0].argmin()]])

        inletPoly = pv.Line(*inlet_pts)
        outlet_pts = np.array([per_y_lower.points[per_y_lower.points[::, 0].argmax()],
                               per_y_upper.points[per_y_upper.points[::, 0].argmax()]])

        outletPoly = pv.Line(*outlet_pts)

        self.yperiodic_low = per_y_lower
        self.yperiodic_high = per_y_upper
        self.inlet = inletPoly
        self.outlet = outletPoly

    def plot_domain(self, figurepath=tempfile.mkdtemp() + "/plot.png"):
        """
        Plot the domain parameters using PyVista.


        Returns:
            pv.Plotter: The PyVista plotter object used for plotting.
        """
        if os.getenv('DISPLAY') is None:
            pv.start_xvfb()  # Start X virtual framebuffer (Xvfb)
        plotter = pv.Plotter(off_screen=True)

        plotter.window_size = 2400, 2400
        # Plot the suction side and pressure side polys
        plotter.add_mesh(self.blade.ss_pv, color='b', show_edges=True)
        plotter.add_mesh(self.blade.ps_pv, color='r', show_edges=True)
        plotter.add_mesh(self.yperiodic_low)
        plotter.add_mesh(self.yperiodic_high)
        plotter.add_mesh(self.inlet)
        plotter.add_mesh(self.outlet)

        plotter.add_axes()
        plotter.view_xy()
        plotter.screenshot(figurepath)
        return figurepath


@dataclass
class CascadeWindTunnelDomain2D:
    casemeta: CaseMeta = CaseMeta(tempfile.mkdtemp())
    blade: Blade2D = None
    domainparams: CascadeWindTunnelDomain2DParameters = None

    pvp0: pv.PolyData = None
    pvp1: pv.PolyData = None
    pvp2: pv.PolyData = None
    pvp3: pv.PolyData = None
    pvp4: pv.PolyData = None
    pvp5: pv.PolyData = None
    pvp6: pv.PolyData = None
    pvp7: pv.PolyData = None
    pvp8: pv.PolyData = None
    pvp9: pv.PolyData = None

    line0: pv.PolyData = None
    line1: pv.PolyData = None
    line2: pv.PolyData = None
    line3: pv.PolyData = None
    line4: pv.PolyData = None
    line5: pv.PolyData = None
    line6: pv.PolyData = None
    line7: pv.PolyData = None
    line8: pv.PolyData = None
    line9: pv.PolyData = None

    ylower_gwk: pv.PolyData = None
    ylower_tailboard: pv.PolyData = None
    yupper_gwk: pv.PolyData = None
    yupper_tailboard: pv.PolyData = None
    inlet: pv.PolyData = None
    outlet: pv.PolyData = None

    def generate_from_cascade_parameters(self, domainparams: CascadeWindTunnelDomain2DParameters, blade: Blade2D):
        # Use params attributes to generate attributes of CascadeDomain2D
        self.blade = blade
        self.gitter_length = domainparams.gittervor + domainparams.gitternach + self.blade.camber_length
        self.tailboard_yd = np.sin(
            np.deg2rad(self.blade.camber_phi + domainparams.gamma + domainparams.tailbeta)) * self.gitter_length
        self.width = domainparams.pitch * (domainparams.nblades + 1)
        zulauf = domainparams.zulauf

        rotation_point = np.array([zulauf, 0, 0])

        blades_rotated, ss_half_idx, ps_half_idx = bladecascade_rotated(domainparams, blade, rotation_point)
        p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = initialdomainpoints(rotation_point, domainparams, blade)
        pvp0, pvp1, pvp2, pvp3, pvp4, pvp5, pvp6, pvp7, pvp8, pvp9 = swivelleddomainpoints(p0, p1, p2, p3, p4, p5, p6,
                                                                                           p7, p8, p9, rotation_point,
                                                                                           domainparams)

        line0, line1, line2, line3, line4, line5, line6, line7, line8, line9 = domainlines(pvp0, pvp1, pvp2, pvp3, pvp4,
                                                                                           pvp5, pvp6, pvp7, pvp8, pvp9,
                                                                                           )

        self.blades = blades_rotated

        self.pvp0 = pvp0
        self.pvp1 = pvp1
        self.pvp2 = pvp2
        self.pvp3 = pvp3
        self.pvp4 = pvp4
        self.pvp5 = pvp5
        self.pvp6 = pvp6
        self.pvp7 = pvp7
        self.pvp8 = pvp8
        self.pvp9 = pvp9

        self.line0 = line0
        self.line1 = line1
        self.line2 = line2
        self.line3 = line3
        self.line4 = line4
        self.line5 = line5
        self.line6 = line6
        self.line7 = line7
        self.line8 = line8
        self.line9 = line9

        self.inlet = pv.Line(pvp5.points[0], pvp0.points[0])
        self.ylower_gwk = pv.Line(pvp0.points[0], pvp1.points[0])
        self.ylower_tailboard = pv.Line(pvp1.points[0], pvp2.points[0])
        self.outlet = pv.Line(pvp2.points[0], pvp3.points[0])
        self.yupper_tailboard = pv.Line(pvp3.points[0], pvp4.points[0])
        self.yupper_gwk = pv.Line(pvp4.points[0], pvp5.points[0])

    def plot_domain(self, figurepath=tempfile.mkdtemp() + "/plot.png"):
        """
        Plot the domain parameters using PyVista.


        Returns:
            pv.Plotter: The PyVista plotter object used for plotting.
        """
        if os.getenv('DISPLAY') is None:
            pv.start_xvfb()  # Start X virtual framebuffer (Xvfb)
        plotter = pv.Plotter(off_screen=True)

        plotter.window_size = 2400, 2400
        # Plot the suction side and pressure side polys
        for b in self.blades:
            plotter.add_mesh(b, color='b', line_width=5)
        plotter.add_mesh(self.ylower_gwk, color="green", label="ylower_gwk", line_width=5)
        plotter.add_mesh(self.ylower_tailboard, color="red", label="ylower_tailboard", line_width=5)
        plotter.add_mesh(self.yupper_gwk, color="green", label="yupper_gwk", line_width=5)
        plotter.add_mesh(self.yupper_tailboard, color="red", label="yupper_tailboard", line_width=5)
        plotter.add_mesh(self.inlet, color="blue", label="inlet", line_width=5)
        plotter.add_mesh(self.outlet, color="blue", label="outlet", line_width=5)
        plotter.add_legend()
        plotter.add_axes()
        plotter.view_xy()
        plotter.screenshot(figurepath)
        return figurepath


def bladecascade_rotated(gwkconfig, blade: Blade2D, rotation_point):
    zulauf = gwkconfig.zulauf
    gittervor = gwkconfig.gittervor
    nblades = gwkconfig.nblades
    width = gwkconfig.pitch * (gwkconfig.nblades + 1)

    pitch = gwkconfig.pitch

    sslengths = lines_from_points(blade.ss_pv.points).compute_cell_sizes()
    sslength = np.sum(sslengths["Length"])
    pslengths = lines_from_points(blade.ps_pv.points).compute_cell_sizes()
    pslength = np.sum(pslengths["Length"])
    ss_half_idx = np.where(np.cumsum(sslengths["Length"]) >= sslength / 2)[0][0]
    ps_half_idx = np.where(np.cumsum(pslengths["Length"]) >= pslength / 2)[0][0] + blade.ite

    bladepoints = [[zulauf + gittervor, i * pitch - width / 2 + pitch, 0] for i in range(nblades)]
    bladeorigin = blade.sortedpoints_pv.points[blade.ile]
    blades = [blade.sortedpoints_pv.copy().translate(-bladeorigin + i) for i in bladepoints]
    blades_rotated = [i.rotate_vector([0, 0, -1], gwkconfig.gamma, rotation_point) for i in blades]

    return blades_rotated, ss_half_idx, ps_half_idx


def initialdomainpoints(rotation_point, gwkconfig, blade, lowres_rwalldist=0.1):
    width = gwkconfig.pitch * (gwkconfig.nblades + 1)
    gitter_length = gwkconfig.gittervor + gwkconfig.gitternach + blade.camber_length
    zulauf = gwkconfig.zulauf
    alpha = gwkconfig.gamma
    tailboard_yd = np.sin(np.deg2rad(blade.camber_phi + gwkconfig.gamma + gwkconfig.tailbeta)) * gitter_length

    p0 = np.array([0, -width / 2, 0])
    p2 = np.array([gitter_length + zulauf, -width / 2, 0])
    p3 = np.array([gitter_length + zulauf, width / 2, 0])
    p5 = np.array([0, width / 2, 0])
    if alpha != 0:
        lower_helper = pv.Line(p0, p2)
        upper_helper = pv.Line(p5, p3)
        lower_helper_rotated = lower_helper.copy().rotate_vector([0, 0, -1], alpha, rotation_point, inplace=True)
        upper_helper_rotated = upper_helper.copy().rotate_vector([0, 0, -1], alpha, rotation_point, inplace=True)
        lower_helper_pa = lower_helper.points[0]
        lower_helper_pb = lower_helper.points[-1]
        upper_helper_pa = upper_helper.points[0]
        upper_helper_pb = upper_helper.points[-1]
        lower_helper_pa_rotated = lower_helper_rotated.points[0]
        lower_helper_pb_rotated = lower_helper_rotated.points[-1]
        upper_helper_pa_rotated = upper_helper_rotated.points[0]
        upper_helper_pb_rotated = upper_helper_rotated.points[-1]
        lower_crossing2d = line_intersection(lower_helper_pa[:2], lower_helper_pb[:2],
                                             lower_helper_pa_rotated[:2], lower_helper_pb_rotated[:2])
        upper_crossing2d = line_intersection(upper_helper_pa[:2], upper_helper_pb[:2],
                                             upper_helper_pa_rotated[:2], upper_helper_pb_rotated[:2])
        p1 = np.array([lower_crossing2d[0], lower_crossing2d[1], 0])
        p4 = np.array([upper_crossing2d[0], upper_crossing2d[1], 0])
        p2 += np.array([0, -tailboard_yd, 0])
        p3 += np.array([0, -tailboard_yd, 0])
    else:
        p1 = p0 + np.array([zulauf, 0, 0])
        p4 = p5 + np.array([zulauf, 0, 0])
        p2 += np.array([0, -tailboard_yd, 0])
        p3 += np.array([0, -tailboard_yd, 0])

    p6 = p0 + (p5 - p0) * (1 - lowres_rwalldist)
    p7 = p1 + (p4 - p1) * (1 - lowres_rwalldist) - vecAbs((p4 - p1) * (2 * lowres_rwalldist)) * np.array([1, 0, 0])
    p8 = p1 + (p4 - p1) * (lowres_rwalldist) - vecAbs((p4 - p1) * (2 * lowres_rwalldist)) * np.array([1, 0, 0])
    p9 = p0 + (p5 - p0) * (lowres_rwalldist)

    return p0, p1, p2, p3, p4, p5, p6, p7, p8, p9


def swivelleddomainpoints(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, rotation_point, gwkconfig):
    alpha = gwkconfig.gamma
    pvp0 = pv.PolyData(p0)
    pvp1 = pv.PolyData(p1)
    pvp2 = pv.PolyData(p2).rotate_vector([0, 0, -1], alpha, rotation_point)
    pvp3 = pv.PolyData(p3).rotate_vector([0, 0, -1], alpha, rotation_point)
    pvp4 = pv.PolyData(p4)
    pvp5 = pv.PolyData(p5)
    pvp6 = pv.PolyData(p6)
    pvp7 = pv.PolyData(p7).rotate_vector([0, 0, -1], alpha, rotation_point)
    pvp8 = pv.PolyData(p8).rotate_vector([0, 0, -1], alpha, rotation_point)
    pvp9 = pv.PolyData(p9)

    return pvp0, pvp1, pvp2, pvp3, pvp4, pvp5, pvp6, pvp7, pvp8, pvp9


def domainlines(pvp0, pvp1, pvp2, pvp3, pvp4, pvp5, pvp6, pvp7, pvp8, pvp9):
    line0 = pv.Line(pvp0.points[0], pvp1.points[0])
    line1 = pv.Line(pvp1.points[0], pvp2.points[0])
    line2 = pv.Line(pvp2.points[0], pvp3.points[0])
    line3 = pv.Line(pvp3.points[0], pvp4.points[0])
    line4 = pv.Line(pvp4.points[0], pvp5.points[0])
    line5 = pv.Line(pvp5.points[0], pvp6.points[0])
    line6 = pv.Line(pvp6.points[0], pvp7.points[0])
    line7 = pv.Line(pvp7.points[0], pvp8.points[0])
    line8 = pv.Line(pvp8.points[0], pvp9.points[0])
    line9 = pv.Line(pvp9.points[0], pvp0.points[0])

    return line0, line1, line2, line3, line4, line5, line6, line7, line8, line9
