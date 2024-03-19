def test_calcmidpassagestreamline():
    from ntrfc.turbo.cascade_case.utils.domain_utils import Blade2D
    from ntrfc.turbo.airfoil_generators.naca_airfoil_creator import naca
    from ntrfc.turbo.cascade_geometry import calcmidpassagestreamline

    import numpy as np
    import pyvista as pv

    naca_code = "6009"
    angle = 10  # deg
    alpha = 1
    res = 240
    xs, ys = naca(naca_code, res, half_cosine_spacing=False)
    points = np.stack([xs, ys, np.zeros(len(xs))]).T
    bladeraw = pv.PolyData(points)
    bladeraw.rotate_z(angle, inplace=True)
    blade = Blade2D(points)
    blade.compute_all_frompoints(alpha=alpha)
    calcmidpassagestreamline(blade.skeletonline_pv.points[::, 0], blade.skeletonline_pv.points[::, 1],
                             blade.beta_le, blade.beta_te, -1, 2, 1)
    blade.plot()
