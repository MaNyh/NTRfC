def test_calcmidpassagestreamline():
    from ntrfc.turbo.pointcloud_methods import extract_geo_paras
    from ntrfc.turbo.airfoil_generators.naca_airfoil_creator import naca
    from ntrfc.turbo.cascade_geometry import calcmidpassagestreamline

    import numpy as np
    import pyvista as pv

    naca_code = "6009"
    angle = 10  # deg
    alpha = 1
    res = 240
    xs, ys = naca(naca_code, res, half_cosine_spacing=False)
    sorted_poly = pv.PolyData(np.stack([xs, ys, np.zeros(len(xs))]).T)
    sorted_poly.rotate_z(angle, inplace=True)

    poly, ps_poly, ss_poly, ind_hk, ind_vk, mids_poly, beta_leading, beta_trailing, camber_angle, alpha = extract_geo_paras(
        sorted_poly, alpha)

    calcmidpassagestreamline(mids_poly.points[::, 0], mids_poly.points[::, 1], beta_leading, beta_trailing, -1, 2, 1)
