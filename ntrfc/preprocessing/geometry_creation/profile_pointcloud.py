from ntrfc.utils.filehandling.read_datafiles import write_pickle, write_yaml_dict
from ntrfc.utils.geometry.pointcloud_methods import extract_geo_paras


import numpy as np
import os

def generate_profile_pointcloud_geometry(settings,basedir):
    ptcloud_profile = settings["ptcloud_profile"]
    ptcloud_profile_unit = settings["ptcloud_profile_unit"]
    alpha = settings["alpha"]

    # =============================================================================
    # Daten Einlesen
    # =============================================================================
    ptcloud_abspath = os.path.join(basedir,ptcloud_profile)
    points = np.loadtxt(ptcloud_abspath)
    unitcoeff = 0
    if ptcloud_profile_unit == "m":
        unitcoeff = 1
    elif ptcloud_profile_unit == "mm":
        unitcoeff = 1 / 1000
    points *= unitcoeff

    # =============================================================================
    # Bestimmung Profilparameter
    # =============================================================================
    sortedPoints, psPoly, ssPoly, ind_vk, ind_hk, midsPoly, beta_meta_01, beta_meta_02, camber_angle = extract_geo_paras(
        points,
        alpha)

    geometry_paras = {"ind_vk":ind_vk,
                    "ind_hk":ind_hk,
                    "beta_meta_01":beta_meta_01,
                    "beta_meta_02":beta_meta_02,
                    "camber_angle":camber_angle}


    geo_dir = os.path.join(basedir,"01_geometry")
    np.savetxt(os.path.join(geo_dir,"sortedPoints.txt"), sortedPoints)
    psPoly.save(os.path.join(geo_dir,"psPoly.vtk"),False)
    ssPoly.save(os.path.join(geo_dir,"ssPoly.vtk"),False)
    midsPoly.save(os.path.join(geo_dir,"midsPoly.vtk"),False)
    write_yaml_dict(os.path.join(geo_dir,"geometry_paras.yml"),geometry_paras)
