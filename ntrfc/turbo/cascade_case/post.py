import tempfile

import numpy as np
from matplotlib import pyplot as plt

from ntrfc.fluid.isentropic import local_isentropic_mach_number
from ntrfc.geometry.plane import massflowave_plane
from ntrfc.math.vectorcalc import vecAbs, vecAngle
from ntrfc.turbo.bladeloading import calc_inflow_cp
from ntrfc.turbo.cascade_case.solution import GenericCascadeCase
from ntrfc.turbo.integrals import avdr


def blade_loading_cp(
    case_instance: GenericCascadeCase,
    pressurevar: str = "pMean",
    densityvar: str = "rhoMean",
    velvar: str = "UMean",
    figpath: str = None
) -> tuple:
    if not figpath:
        figpath = tempfile.mkdtemp() + "/blade_loading_cp.png"

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
        #minmax norm ps_xc

        ps_xc[idx] = (pts1[0] - px_min) / cax_len
        ps_cp[idx] = calc_inflow_cp(psmeshpoints.point_data[pressurevar][idx], pt1, p1)

    ss_xc = np.zeros(ssmeshpoints.number_of_points)
    ss_cp = np.zeros(ssmeshpoints.number_of_points)

    for idx, pts1 in enumerate(ssmeshpoints.points):
        ss_xc[idx] = (pts1[0] - px_min) / cax_len
        ss_cp[idx] = calc_inflow_cp(ssmeshpoints.point_data[pressurevar][idx], pt1, p1)

    plt.figure()
    plt.title("blade loading")
    plt.scatter(ss_xc, ss_cp, label="suction side")
    plt.scatter(ps_xc, ps_cp, label="pressure side")
    plt.xlabel("$x/c_{ax}$")
    plt.ylabel("$c_{p}$")
    plt.grid()
    plt.legend()
    plt.savefig(figpath)
    plt.close()

    return ss_xc, ss_cp, ps_xc, ps_cp, figpath


def blade_loading_mais(case_instance, pressurevar="pMean", densityvar="rhoMean", velvar="UMean",
                       figpath=tempfile.mkdtemp() + "/blade_loading_mais.png", kappa=1.4):
    camber_length = case_instance.blade.camber_length

    inlet = case_instance.mesh_dict["inlet"]
    inlet["u"] = inlet[velvar][::, 0]
    inlet["v"] = inlet[velvar][::, 1]
    inlet["w"] = inlet[velvar][::, 2]
    # p1 = massflowave_plane(inlet, valname=pressurevar, rhoname=densityvar, velocityname=velvar)
    # rho = massflowave_plane(inlet, valname=densityvar, rhoname=densityvar, velocityname=velvar)
    # u = massflowave_plane(inlet, valname="u", rhoname=densityvar, velocityname=velvar)
    # v = massflowave_plane(inlet, valname="v", rhoname=densityvar, velocityname=velvar)
    # w = massflowave_plane(inlet, valname="w", rhoname=densityvar, velocityname=velvar)
    # U = vecAbs([u, v, w])
    # pt1 = p1 + 1 / 2 * rho * U ** 2

    ssmeshpoints = case_instance.blade.ss_pv
    psmeshpoints = case_instance.blade.ps_pv

    ps_xc = np.zeros(psmeshpoints.number_of_points)
    ps_cp = np.zeros(psmeshpoints.number_of_points)

    pressure_inlet = massflowave_plane(case_instance.mesh_dict["inlet"], valname=pressurevar, rhoname=densityvar,
                                       velocityname=velvar)
    density_inlet = massflowave_plane(case_instance.mesh_dict["inlet"], valname=densityvar, rhoname=densityvar,
                                      velocityname=velvar)
    case_instance.mesh_dict["inlet"]["u"] = case_instance.mesh_dict["inlet"][velvar][::, 0]
    case_instance.mesh_dict["inlet"]["v"] = case_instance.mesh_dict["inlet"][velvar][::, 1]
    case_instance.mesh_dict["inlet"]["w"] = case_instance.mesh_dict["inlet"][velvar][::, 2]
    velocity_inlet_u = massflowave_plane(case_instance.mesh_dict["inlet"], valname="u", rhoname=densityvar,
                                         velocityname=velvar)
    velocity_inlet_v = massflowave_plane(case_instance.mesh_dict["inlet"], valname="v", rhoname=densityvar,
                                         velocityname=velvar)
    velocity_inlet_w = massflowave_plane(case_instance.mesh_dict["inlet"], valname="w", rhoname=densityvar,
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
    plt.scatter(ss_xc, ss_cp, label="suction side")
    plt.scatter(ps_xc, ps_cp, label="pressure side")
    plt.xlabel("$x/c_{ax}$")
    plt.ylabel("$Ma_{is}$")
    plt.grid()
    plt.legend()
    plt.savefig(figpath)
    plt.close()
    return ss_xc, ss_cp, ps_xc, ps_cp, figpath


def compute_avdr_inout_massave(case_instance, densityvar="rhoMean", velvar="UMean", ):
    case_instance.mesh_dict["inlet"]["u"] = case_instance.mesh_dict["inlet"][velvar][::, 0]
    case_instance.mesh_dict["inlet"]["v"] = case_instance.mesh_dict["inlet"][velvar][::, 1]
    case_instance.mesh_dict["inlet"]["w"] = case_instance.mesh_dict["inlet"][velvar][::, 2]

    case_instance.mesh_dict["outlet"]["u"] = case_instance.mesh_dict["outlet"][velvar][::, 0]
    case_instance.mesh_dict["outlet"]["v"] = case_instance.mesh_dict["outlet"][velvar][::, 1]
    case_instance.mesh_dict["outlet"]["w"] = case_instance.mesh_dict["outlet"][velvar][::, 2]
    rho_1 = massflowave_plane(case_instance.mesh_dict["inlet"], valname=densityvar, rhoname=densityvar,
                              velocityname=velvar)
    mag_u_1 = vecAbs(
        np.array([massflowave_plane(case_instance.mesh_dict["inlet"], "u", rhoname=densityvar, velocityname=velvar),
                  massflowave_plane(case_instance.mesh_dict["inlet"], "v", rhoname=densityvar, velocityname=velvar),
                  massflowave_plane(case_instance.mesh_dict["inlet"], "w", rhoname=densityvar, velocityname=velvar)]))
    U_1 = np.stack(
        [massflowave_plane(case_instance.mesh_dict["inlet"], "u", rhoname=densityvar, velocityname=velvar),
         massflowave_plane(case_instance.mesh_dict["inlet"], "v", rhoname=densityvar, velocityname=velvar),
         massflowave_plane(case_instance.mesh_dict["inlet"], "w", rhoname=densityvar, velocityname=velvar)])
    beta_1 = vecAngle(U_1, np.array([1, 0, 0]))
    rho_2 = massflowave_plane(case_instance.mesh_dict["outlet"], densityvar, rhoname=densityvar, velocityname=velvar)
    U_2 = np.stack(
        [massflowave_plane(case_instance.mesh_dict["outlet"], "u", rhoname=densityvar, velocityname=velvar),
         massflowave_plane(case_instance.mesh_dict["outlet"], "v", rhoname=densityvar, velocityname=velvar),
         massflowave_plane(case_instance.mesh_dict["outlet"], "w", rhoname=densityvar, velocityname=velvar)])
    mag_u_2 = vecAbs(np.array(
        [massflowave_plane(case_instance.mesh_dict["outlet"], "u", rhoname=densityvar, velocityname=velvar),
         massflowave_plane(case_instance.mesh_dict["outlet"], "v", rhoname=densityvar, velocityname=velvar),
         massflowave_plane(case_instance.mesh_dict["outlet"], "w", rhoname=densityvar, velocityname=velvar)]))
    beta_2 = vecAngle(U_2, np.array([1, 0, 0]))
    case_instance.avdr = avdr(rho_1, mag_u_1, beta_1, rho_2, mag_u_2, beta_2)
    return case_instance.avdr
