import tempfile

import numpy as np
import pyvista as pv
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

import ntrfc
from ntrfc.fluid.isentropic import local_isentropic_mach_number
from ntrfc.geometry.plane import massflowave_plane
from ntrfc.math.vectorcalc import vecAbs, vecAngle, vecAbs_list
from ntrfc.meshquality.aspect_ratio import compute_cell_aspect_ratios
from ntrfc.meshquality.meshexpansion import compute_expansion_factors
from ntrfc.meshquality.nondimensionals import calc_dimensionless_gridspacing
from ntrfc.meshquality.skewness import compute_cell_skewness
from ntrfc.meshquality.standards import quality_definitions
from ntrfc.turbo.bladeloading import calc_inflow_cp
from ntrfc.turbo.cascade_case.solution import GenericCascadeCase, Blade2D
from ntrfc.turbo.integrals import avdr


def cascade_case_meshquality(case_instance: GenericCascadeCase,
                             pdfpath_qualityreport: str = None,
                             figpath_qualitystats: str = None,
                             figpath_meshslice: str = None,
                             figpath_badcells: str = None) -> str:
    if not pdfpath_qualityreport:
        pdfpath_qualityreport = tempfile.mkdtemp() + "/quality_report.pdf"
    if not figpath_qualitystats:
        figpath_qualitystats = tempfile.mkdtemp() + "/meshquality.png"
    if not figpath_meshslice:
        figpath_meshslice = tempfile.mkdtemp() + "/meshslice.png"
    if not figpath_badcells:
        figpath_badcells = tempfile.mkdtemp() + "/badcells.png"

    grid = case_instance.mesh_dict["fluid"]
    print("[ntrfc info] computing aspect ratios...")
    aspect_ratios = compute_cell_aspect_ratios(grid)
    print("[ntrfc info] computing skewnesses...")
    skewnesses = compute_cell_skewness(grid)
    print("[ntrfc info] computing expansion factors...")
    expansion_factors = compute_expansion_factors(grid)

    grid["aspect_ratios"] = aspect_ratios
    grid["skewnesses"] = skewnesses
    grid["expansion_factors"] = expansion_factors

    plot_meshstats_histograms(aspect_ratios, expansion_factors, figpath_qualitystats, skewnesses)

    plot_gridslice(figpath_meshslice, grid)

    plot_badcells(aspect_ratios, expansion_factors, figpath_badcells, grid, skewnesses)

    images = [
        Image.open(f)
        for f in [figpath_meshslice, figpath_qualitystats, figpath_badcells]
    ]
    images[0].save(
        pdfpath_qualityreport, "PDF", resolution=300.0, save_all=True, append_images=images[1:]
    )

    return figpath_qualitystats, figpath_meshslice, figpath_badcells, pdfpath_qualityreport


def plot_badcells(aspect_ratios, expansion_factors, figpath_badcells, grid, skewnesses):
    ok_aspect_ratios = quality_definitions["AspectRatio"]["ok"]
    ok_skewnesses = quality_definitions["Skewness"]["ok"]
    ok_expansion_factors = quality_definitions["MeshExpansion"]["ok"]
    ok_aspect_ratios_cells_mask = (aspect_ratios > min(ok_aspect_ratios)) & (aspect_ratios < max(ok_aspect_ratios))
    ok_skewnesses_cells_mask = (skewnesses > min(ok_skewnesses)) & (skewnesses < max(ok_skewnesses))
    ok_expansion_factors_cells_mask = (expansion_factors > min(ok_expansion_factors)) & (
        expansion_factors < max(ok_expansion_factors))
    bad_aspect_ratio_cells_mask = aspect_ratios >= max(ok_aspect_ratios)
    bad_skewness_cells_mask = skewnesses >= max(ok_skewnesses)
    bad_expansion_factors_cells_mask = expansion_factors >= max(ok_expansion_factors)
    ok_skewnesses_cells = grid.extract_cells(ok_skewnesses_cells_mask)
    ok_aspect_ratios_cells = grid.extract_cells(ok_aspect_ratios_cells_mask)
    ok_expansion_factors_cells = grid.extract_cells(ok_expansion_factors_cells_mask)
    bad_skewness_cells = grid.extract_cells(bad_skewness_cells_mask)
    bad_aspect_ratio_cells = grid.extract_cells(bad_aspect_ratio_cells_mask)
    bad_expansion_factors_cells = grid.extract_cells(bad_expansion_factors_cells_mask)
    ok_color = "cyan"
    bad_color = "red"
    p = pv.Plotter(off_screen=True)
    if ok_aspect_ratios_cells.n_cells > 0:
        p.add_mesh(ok_aspect_ratios_cells, color=ok_color, opacity=0.2, show_edges=True)
    if ok_skewnesses_cells.n_cells > 0:
        p.add_mesh(ok_skewnesses_cells, color=ok_color, opacity=0.2, show_edges=True)
    if ok_expansion_factors_cells.n_cells > 0:
        p.add_mesh(ok_expansion_factors_cells, color=ok_color, opacity=0.2, show_edges=True)
    if bad_aspect_ratio_cells.n_cells > 0:
        p.add_mesh(bad_aspect_ratio_cells, color=bad_color, opacity=0.5, show_edges=True)
    if bad_skewness_cells.n_cells > 0:
        p.add_mesh(bad_skewness_cells, color=bad_color, opacity=0.5, show_edges=True)
    if bad_expansion_factors_cells.n_cells > 0:
        p.add_mesh(bad_expansion_factors_cells, color=bad_color, opacity=0.5, show_edges=True)
    p.view_xy()
    p.show(screenshot=figpath_badcells)


def plot_gridslice(figpath_meshslice, grid):
    # Plot the mesh slice
    p = pv.Plotter(off_screen=True)
    p.add_mesh(grid.slice(normal="z"), color="grey", opacity=0.5, show_edges=True)
    p.view_xy()
    p.show(screenshot=figpath_meshslice)


def plot_meshstats_histograms(aspect_ratios, expansion_factors, figpath_qualitystats, skewnesses):
    # Plot the quality stats
    fig, axs = plt.subplots(3, 1, figsize=(15, 10))
    axs[0].hist(aspect_ratios, bins=64, color="blue", alpha=0.7)
    axs[0].set_title("Aspect Ratios")
    axs[0].set_xlabel("Aspect Ratio")
    axs[0].set_ylabel("Frequency")
    axs[0].grid()
    axs[1].hist(skewnesses, bins=64, color="red", alpha=0.7)
    axs[1].set_title("Skewnesses")
    axs[1].set_xlabel("Skewness")
    axs[1].set_ylabel("Frequency")
    axs[1].grid()
    axs[2].hist(expansion_factors, bins=64, color="green", alpha=0.7)
    axs[2].set_title("Expansion Factors")
    axs[2].set_xlabel("Expansion Factor")
    axs[2].set_ylabel("Frequency")
    axs[2].grid()
    # Show texts
    axs[0].text(0.9, 0.5,
                f"min {np.round(np.min(aspect_ratios), 2)}\n mean: {np.round(np.mean(aspect_ratios), 2)}\n max:{np.round(np.max(aspect_ratios), 2)}",
                horizontalalignment='right', verticalalignment='center', transform=axs[0].transAxes)
    axs[1].text(0.9, 0.5,
                f"min {np.round(np.min(skewnesses), 2)}\n mean: {np.round(np.mean(skewnesses), 2)}\n max:{np.round(np.max(skewnesses), 2)}",
                horizontalalignment='right', verticalalignment='center', transform=axs[1].transAxes)
    axs[2].text(0.9, 0.5,
                f"min {np.round(np.min(expansion_factors), 2)}\n mean: {np.round(np.mean(expansion_factors), 2)}\n max:{np.round(np.max(expansion_factors), 2)}",
                horizontalalignment='right', verticalalignment='center', transform=axs[2].transAxes)
    plt.tight_layout()
    plt.savefig(figpath_qualitystats)


def cascade_case_tux(case_instance: GenericCascadeCase, res: int = 200, figpath: str = None) -> str:
    if not figpath:
        figpath = tempfile.mkdtemp() + "/cascade_case_tux.png"

    grid = case_instance.mesh_dict["fluid"]
    umeanname = case_instance.case_meta.meanvelocity_name
    rhomeanname = case_instance.case_meta.meandensity_name
    kmeanname = case_instance.case_meta.meanturbulentkineticenergy_name
    x0 = grid.bounds[0]
    x1 = grid.bounds[1]
    xs = np.linspace(x0 + 1e-6, x1 - 1e-6, res)
    uabs = vecAbs_list(grid.cell_data[umeanname])
    grid["uabs"] = uabs
    tus = []
    for x in tqdm(xs):
        slicex = grid.slice(normal="x", origin=(x, 0, 0))
        uabs = massflowave_plane(slicex, valname="uabs", rhoname=rhomeanname, velocityname=umeanname)
        tke = massflowave_plane(slicex, valname=kmeanname, rhoname=rhomeanname, velocityname=umeanname)
        tus.append(np.sqrt(2 / 3 * tke) / uabs)

    fig = plt.figure()
    plt.plot(xs, tus)
    plt.xlabel("x")
    plt.ylabel("Tu")
    plt.title("Turbulence intensity profile")
    plt.grid()
    fig.text(0.05, 0.01, f"ntrfc.meshquality.nondimensionals@{ntrfc.__version__}", fontsize=16, color="grey", alpha=0.5)

    plt.savefig(figpath)
    plt.close()

    return figpath


def blade_deltas(
    case_instance: GenericCascadeCase,
    figpath: str = None,
) -> str:
    if not figpath:
        figpath = tempfile.mkdtemp() + "/blade_deltas.png"

    grid = case_instance.mesh_dict["fluid"]
    blade = case_instance.mesh_dict["blade"]
    velocity_arrayname = case_instance.case_meta.meanvelocity_name
    density_arrayname = case_instance.case_meta.meandensity_name
    mu_ref = case_instance.case_meta.dynamic_viscosity

    dimless_gridspacings = calc_dimensionless_gridspacing(grid, [blade], velocity_arrayname, density_arrayname, mu_ref)
    delta_cells_one = dimless_gridspacings.slice(normal="z")
    delta_cc_one = delta_cells_one.cell_centers()

    blade = Blade2D(delta_cc_one)
    blade.compute_all_frompoints()

    pspoly_one = blade.ps_pv
    sspoly_one = blade.ss_pv

    ps_delta_one = pspoly_one.interpolate(delta_cc_one)
    ss_delta_one = sspoly_one.interpolate(delta_cc_one)

    x0_one = ps_delta_one.points[0][0]
    x1_one = ps_delta_one.points[-1][0]

    ps_rc_one = (ps_delta_one.points[::, 0] - x0_one) / (x1_one - x0_one)
    ss_rc_one = (ss_delta_one.points[::, 0] - x0_one) / (x1_one - x0_one)

    fig, axes = plt.subplots(3, 1, figsize=(15, 10))

    axes[0].grid(axis="y")
    axes[0].plot(ss_rc_one * -1, ss_delta_one.point_data["DeltaXPlus"], color='c', label='ps')
    axes[0].plot(ps_rc_one, ps_delta_one.point_data["DeltaXPlus"], color='c', linestyle='dashed', label='ss')

    axes[0].set_title('DeltaXPlus')
    axes[0].set_xlabel(r'$x/c_{ax}$')
    axes[0].set_ylabel(r'$\Delta \xi^{+}$', rotation=0, size=15, labelpad=20)
    # axes[0].set_ylim(0,60)
    axes[0].legend()

    constrained_y_ticks_2 = [1, 2, 3, 4]
    axes[1].set_yticks(constrained_y_ticks_2)
    axes[1].grid(axis="y")
    axes[1].plot(ss_rc_one * -1, ss_delta_one.point_data["DeltaYPlus"], color='c', label='ps')
    axes[1].plot(ps_rc_one, ps_delta_one.point_data["DeltaYPlus"], color='c', linestyle='dashed', label='ss')

    axes[1].set_title('DeltaYPlus')
    axes[1].set_xlabel(r'$x/c_{ax}$')
    axes[1].set_ylabel(r'$\Delta \eta^{+}$', rotation=0, size=15, labelpad=20)
    # axes[1].set_ylim(0,4)

    constrained_y_ticks_3 = [25, 50]
    axes[2].set_yticks(constrained_y_ticks_3)
    axes[2].grid(axis="y")

    axes[2].plot(ss_rc_one * -1, ss_delta_one.point_data["DeltaZPlus"], color='c', label='ps')
    axes[2].plot(ps_rc_one, ps_delta_one.point_data["DeltaZPlus"], color='c', linestyle='dashed', label='ss')

    axes[2].set_title('DeltaZPlus')
    axes[2].set_xlabel(r'$x/c_{ax}$')
    # axes[2].set_ylim(0,60)
    axes[2].set_ylabel(r'$\Delta \zeta^{+}$', rotation=0, size=15, labelpad=20)
    fig.text(0.05, 0.01, f"ntrfc.meshquality.nondimensionals@{ntrfc.__version__}", fontsize=16, color="grey", alpha=0.5)

    plt.tight_layout()

    # Show the plot
    plt.savefig(figpath)

    return figpath


def blade_loading_cp(
    case_instance: GenericCascadeCase,
    figpath: str = None
) -> tuple:
    if not figpath:
        figpath = tempfile.mkdtemp() + "/blade_loading_cp.png"

    if not case_instance.blade:
        raise ValueError("blade not initialized, run case.set_bladeslice_midz() first")

    pressurevar = case_instance.case_meta.meanpressure_name  # "pMean",
    densityvar = case_instance.case_meta.meandensity_name  # "rhoMean",
    velvar = case_instance.case_meta.meanvelocity_name

    inlet = case_instance.mesh_dict["inlet"]
    inlet["u"] = inlet[velvar][::, 0]
    inlet["v"] = inlet[velvar][::, 1]
    inlet["w"] = inlet[velvar][::, 2]
    p1 = massflowave_plane(inlet, valname=pressurevar, rhoname=densityvar, velocityname=velvar)
    rho = massflowave_plane(inlet, valname=densityvar, rhoname=densityvar, velocityname=velvar)
    u = massflowave_plane(inlet, valname="u", rhoname=densityvar, velocityname=velvar)
    v = massflowave_plane(inlet, valname="v", rhoname=densityvar, velocityname=velvar)
    w = massflowave_plane(inlet, valname="w", rhoname=densityvar, velocityname=velvar)
    U = vecAbs([u, v, w])

    px_min = np.min(case_instance.blade.ps_pv.points[:, 0])
    px_max = np.max(case_instance.blade.ps_pv.points[:, 0])
    cax_len = px_max - px_min

    pt1 = p1 + 1 / 2 * rho * U ** 2

    ssmeshpoints = case_instance.blade.ss_pv
    psmeshpoints = case_instance.blade.ps_pv

    ps_xc = np.zeros(psmeshpoints.number_of_points)
    ps_cp = np.zeros(psmeshpoints.number_of_points)

    for idx, pts1 in enumerate(psmeshpoints.points):
        ps_xc[idx] = (pts1[0] - px_min) / cax_len
        ps_cp[idx] = calc_inflow_cp(psmeshpoints.point_data[pressurevar][idx], pt1, p1)

    ss_xc = np.zeros(ssmeshpoints.number_of_points)
    ss_cp = np.zeros(ssmeshpoints.number_of_points)

    for idx, pts1 in enumerate(ssmeshpoints.points):
        ss_xc[idx] = (pts1[0] - px_min) / cax_len
        ss_cp[idx] = calc_inflow_cp(ssmeshpoints.point_data[pressurevar][idx], pt1, p1)

    plt.figure()
    plt.title("blade loading")
    plt.plot(ss_xc, ss_cp, label="suction side")
    plt.plot(ps_xc, ps_cp, label="pressure side")
    plt.xlabel("$x/c_{ax}$")
    plt.ylabel("$c_{p}$")
    plt.grid()
    plt.legend()
    plt.savefig(figpath)
    plt.close()

    return ss_xc, ss_cp, ps_xc, ps_cp, figpath


def blade_loading_absolute(
    case_instance: GenericCascadeCase,
    figpath: str = None
) -> tuple:
    if not figpath:
        figpath = tempfile.mkdtemp() + "/blade_loading_abs.png"

    if not case_instance.blade:
        raise ValueError("blade not initialized, run case.set_bladeslice_midz() first")

    pressurevar = case_instance.case_meta.meanpressure_name
    velvar = case_instance.case_meta.meanvelocity_name

    inlet = case_instance.mesh_dict["inlet"]
    inlet["u"] = inlet[velvar][::, 0]
    inlet["v"] = inlet[velvar][::, 1]
    inlet["w"] = inlet[velvar][::, 2]

    px_min = np.min(case_instance.blade.ps_pv.points[:, 0])
    px_max = np.max(case_instance.blade.ps_pv.points[:, 0])
    cax_len = px_max - px_min

    ssmeshpoints = case_instance.blade.ss_pv
    psmeshpoints = case_instance.blade.ps_pv

    ps_xc = np.zeros(psmeshpoints.number_of_points)
    ps_cp = np.zeros(psmeshpoints.number_of_points)

    for idx, pts1 in enumerate(psmeshpoints.points):
        ps_xc[idx] = (pts1[0] - px_min) / cax_len
        ps_cp[idx] = psmeshpoints.point_data[pressurevar][idx]

    ss_xc = np.zeros(ssmeshpoints.number_of_points)
    ss_cp = np.zeros(ssmeshpoints.number_of_points)

    for idx, pts1 in enumerate(ssmeshpoints.points):
        ss_xc[idx] = (pts1[0] - px_min) / cax_len
        ss_cp[idx] = ssmeshpoints.point_data[pressurevar][idx]

    plt.figure()
    plt.title("blade loading")
    plt.plot(ss_xc, ss_cp, label="suction side")
    plt.plot(ps_xc, ps_cp, label="pressure side")
    plt.xlabel("$x/c_{ax}$")
    plt.ylabel("$p$")
    plt.grid()
    plt.legend()
    plt.savefig(figpath)
    plt.close()

    return ss_xc, ss_cp, ps_xc, ps_cp, figpath


def blade_loading_mais(case_instance,
                       figpath=tempfile.mkdtemp() + "/blade_loading_mais.png"):
    pressurevar = case_instance.case_meta.meanpressure_name
    densityvar = case_instance.case_meta.meandensity_name
    velvar = case_instance.case_meta.meanvelocity_name
    kappa = case_instance.case_meta.kappa

    camber_length = case_instance.blade.camber_length

    inlet = case_instance.mesh_dict["inlet"]

    inlet["u"] = inlet[velvar][::, 0]
    inlet["v"] = inlet[velvar][::, 1]
    inlet["w"] = inlet[velvar][::, 2]

    ssmeshpoints = case_instance.blade.ss_pv
    psmeshpoints = case_instance.blade.ps_pv

    ps_xc = np.zeros(psmeshpoints.number_of_points)
    ps_cp = np.zeros(psmeshpoints.number_of_points)

    pressure_inlet = massflowave_plane(inlet, valname=pressurevar, rhoname=densityvar,
                                       velocityname=velvar)
    density_inlet = massflowave_plane(inlet, valname=densityvar, rhoname=densityvar,
                                      velocityname=velvar)
    inlet["u"] = inlet[velvar][::, 0]
    inlet["v"] = inlet[velvar][::, 1]
    inlet["w"] = inlet[velvar][::, 2]
    velocity_inlet_u = massflowave_plane(inlet, valname="u", rhoname=densityvar,
                                         velocityname=velvar)
    velocity_inlet_v = massflowave_plane(inlet, valname="v", rhoname=densityvar,
                                         velocityname=velvar)
    velocity_inlet_w = massflowave_plane(inlet, valname="w", rhoname=densityvar,
                                         velocityname=velvar)
    velocity_inlet = vecAbs(np.array([velocity_inlet_u, velocity_inlet_v, velocity_inlet_w]))
    totalpressure_inlet = pressure_inlet + 0.5 * density_inlet * velocity_inlet ** 2

    for idx, pts1 in enumerate(psmeshpoints.points):
        ps_xc[idx] = pts1[0] / camber_length
        bladepressure = psmeshpoints.point_data[pressurevar][idx]
        ps_cp[idx] = local_isentropic_mach_number(kappa, bladepressure, totalpressure_inlet)

    ss_xc = np.zeros(ssmeshpoints.number_of_points)
    ss_cp = np.zeros(ssmeshpoints.number_of_points)

    for idx, pts1 in enumerate(ssmeshpoints.points):
        ss_xc[idx] = pts1[0] / camber_length
        bladepressure = ssmeshpoints.point_data[pressurevar][idx]
        ss_cp[idx] = local_isentropic_mach_number(kappa, bladepressure, totalpressure_inlet)

    plt.figure()
    plt.title("blade loading")
    plt.plot(ss_xc, ss_cp, label="suction side")
    plt.plot(ps_xc, ps_cp, label="pressure side")
    plt.xlabel("$x/c_{ax}$")
    plt.ylabel("$Ma_{is}$")
    plt.grid()
    plt.legend()
    plt.savefig(figpath)
    plt.close()
    return ss_xc, ss_cp, ps_xc, ps_cp, figpath


def compute_avdr_inout_massave(case_instance):
    densityvar = case_instance.case_meta.meandensity_name
    velvar = case_instance.case_meta.meanvelocity_name

    inlet = case_instance.mesh_dict["inlet"]
    inlet["u"] = inlet[velvar][::, 0]
    inlet["v"] = inlet[velvar][::, 1]
    inlet["w"] = inlet[velvar][::, 2]

    outlet = case_instance.mesh_dict["outlet"]
    outlet["u"] = outlet[velvar][::, 0]
    outlet["v"] = outlet[velvar][::, 1]
    outlet["w"] = outlet[velvar][::, 2]
    rho_1 = massflowave_plane(inlet, valname=densityvar, rhoname=densityvar,
                              velocityname=velvar)
    mag_u_1 = vecAbs(
        np.array([massflowave_plane(inlet, "u", rhoname=densityvar, velocityname=velvar),
                  massflowave_plane(inlet, "v", rhoname=densityvar, velocityname=velvar),
                  massflowave_plane(inlet, "w", rhoname=densityvar, velocityname=velvar)]))
    U_1 = np.stack(
        [massflowave_plane(inlet, "u", rhoname=densityvar, velocityname=velvar),
         massflowave_plane(inlet, "v", rhoname=densityvar, velocityname=velvar),
         massflowave_plane(inlet, "w", rhoname=densityvar, velocityname=velvar)])
    beta_1 = vecAngle(U_1, np.array([1, 0, 0]))
    rho_2 = massflowave_plane(outlet, densityvar, rhoname=densityvar, velocityname=velvar)
    U_2 = np.stack(
        [massflowave_plane(outlet, "u", rhoname=densityvar, velocityname=velvar),
         massflowave_plane(outlet, "v", rhoname=densityvar, velocityname=velvar),
         massflowave_plane(outlet, "w", rhoname=densityvar, velocityname=velvar)])
    mag_u_2 = vecAbs(np.array(
        [massflowave_plane(outlet, "u", rhoname=densityvar, velocityname=velvar),
         massflowave_plane(outlet, "v", rhoname=densityvar, velocityname=velvar),
         massflowave_plane(outlet, "w", rhoname=densityvar, velocityname=velvar)]))
    beta_2 = vecAngle(U_2, np.array([1, 0, 0]))
    case_instance.avdr = avdr(rho_1, mag_u_1, beta_1, rho_2, mag_u_2, beta_2)
    return case_instance.avdr


def cascadecase_contour(case_instance, varname, figpath=None):
    if not figpath:
        figpath = tempfile.mkdtemp() + f"/contour_{varname}.png"

    grid = case_instance.mesh_dict["fluid"]
    slicez = grid.slice(normal="z")
    p = pv.Plotter(off_screen=True)
    p.add_mesh(slicez, scalars=case_instance.case_meta.casevariables(varname), show_scalar_bar=True)
    p.view_xy()
    p.show(screenshot=figpath)

    return figpath
