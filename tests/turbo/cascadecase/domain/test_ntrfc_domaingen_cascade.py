import os

ON_CI = 'CI' in os.environ


def test_cascade_2d_domain():
    import pyvista as pv
    import numpy as np
    from ntrfc.turbo.cascade_case.domain import CascadeDomain2D
    from ntrfc.turbo.cascade_case.utils.domain_utils import CascadeDomain2DParameters, Blade2D
    from ntrfc.turbo.airfoil_generators.naca_airfoil_creator import naca
    xs, ys = naca("6510", 256)
    points = pv.PolyData(np.stack([xs, ys, np.zeros(len(xs))]).T)
    alpha = 1
    blade = Blade2D(points)
    blade.compute_all_frompoints(alpha=alpha)
    domainparas = CascadeDomain2DParameters(xinlet=-3, xoutlet=4, pitch=2, blade_yshift=0.1)
    domain2d = CascadeDomain2D()
    domain2d.generate_from_cascade_parameters(domainparas, blade)
    domain2d.plot_domain()
