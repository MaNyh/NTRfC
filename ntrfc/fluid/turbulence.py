import numpy as np


def calcTu(tke, Uabs):
    # tke: turbulente kinetische Energie
    # Uabs: zeitlich gemittelte lokale Absolutgeschinwidgkeit

    Tu = np.sqrt(2.0 / 3.0 * tke) / max(1e-9, Uabs)

    return Tu


def calcTkeByTu(Tu, Uabs):
    Tke = (Tu * Uabs) ** 2 * (3 / 2.0)
    return Tke


def calcTke(u_2, v_2, w_2):
    # u_2: u-fluktuationen quadriert und gemittelt
    # v_2: v-fluktuationen quadriert und gemittelt
    # w_2: w-fluktuationen quadriert und gemittelt
    tke = (0.5 * (u_2 + v_2 + w_2))
    return tke


def calcFluc(velo):
    mean_velo = np.mean(velo)

    velo = np.array(velo)

    return velo - mean_velo


def calcRey(u, v, w):
    u_ = calcFluc(u)
    v_ = calcFluc(v)
    w_ = calcFluc(w)

    u_u_ = np.multiply(u_, u_)
    v_v_ = np.multiply(v_, v_)
    w_w_ = np.multiply(w_, w_)
    u_v_ = np.multiply(u_, v_)
    u_w_ = np.multiply(u_, w_)
    v_w_ = np.multiply(v_, w_)

    mean_u_u_ = np.mean(u_u_)
    mean_v_v_ = np.mean(v_v_)
    mean_w_w_ = np.mean(w_w_)
    mean_u_v_ = np.mean(u_v_)
    mean_u_w_ = np.mean(u_w_)
    mean_v_w_ = np.mean(v_w_)

    return [mean_u_u_, mean_v_v_, mean_w_w_, mean_u_v_, mean_u_w_, mean_v_w_]
