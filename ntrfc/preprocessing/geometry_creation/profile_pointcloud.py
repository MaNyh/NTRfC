from ntrfc.utils.filehandling.read_datafiles import write_pickle, write_yaml_dict
from ntrfc.utils.geometry.pointcloud_methods import extract_geo_paras


import numpy as np
import os
import matplotlib.pyplot as plt

def generate_profile_pointcloud_geometry(basedir,**params):

    ptcloud_profile = params["ptcloud_profile"]
    ptcloud_profile_unit = params["ptcloud_profile_unit"]
    alpha = params["alpha"]

    # =============================================================================
    # Daten Einlesen
    # =============================================================================
    ptcloud_abspath = os.path.join(basedir,"00_resources",ptcloud_profile)
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

    geometry_paras = {"ind_vk":int(ind_vk),
                    "ind_hk":int(ind_hk),
                    "beta_meta_01":float(beta_meta_01),
                    "beta_meta_02":float(beta_meta_02),
                    "camber_angle":float(camber_angle)}


    geo_dir = os.path.join(basedir,"01_profile")
    np.savetxt(os.path.join(geo_dir,"sortedPoints.txt"), sortedPoints)
    psPoly.save(os.path.join(geo_dir,"psPoly.vtk"),False)
    ssPoly.save(os.path.join(geo_dir,"ssPoly.vtk"),False)
    midsPoly.save(os.path.join(geo_dir,"midsPoly.vtk"),False)
    write_yaml_dict(os.path.join(geo_dir,"geometry_paras.yml"),geometry_paras)

    #todo: here, we should also plot the geometry-parameters (chord, angles, ...)
    plt.figure()
    plt.plot(psPoly.points[::,0],psPoly.points[::,1], color="#6c3376")
    plt.plot(ssPoly.points[::,0],ssPoly.points[::,1], color="#FF2211")
    plt.plot(midsPoly.points[::,0],midsPoly.points[::,1], color="#FF22CC")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(os.path.join(geo_dir,'profile.pdf'))

