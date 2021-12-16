from ntrfc.utils.geometry.pointcloud_methods import ex

def generate_profile_pointcloud_geometry(settings):
    ptcloud_profile = config["blade"]["ptcloud_profile"]
    ptcloud_profile_unit = config["blade"]["ptcloud_profile_unit"]
    alpha = config["blade"]["alpha"]

    # =============================================================================
    # Daten Einlesen
    # =============================================================================
    points = np.loadtxt(ptcloud_profile)
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
        alpha,
        verbose)
