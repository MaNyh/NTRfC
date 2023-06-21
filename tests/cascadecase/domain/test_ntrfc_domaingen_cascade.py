def test_cascade_2d_domain():
    import pyvista as pv
    import numpy as np
    from ntrfc.cascade_case.utils.domain_utils import DomainParameters
    from ntrfc.cascade_case.domain import CascadeDomain2D
    from ntrfc.turbo.airfoil_generators.naca_airfoil_creator import naca
    xs, ys = naca("6510", 256)
    points = pv.PolyData(np.stack([xs, ys, np.zeros(len(xs))]).T)
    alpha = 1
    domainparas = DomainParameters()
    domainparas.generate_params_by_pointcloud(points, alpha=alpha)
    domainparas.xinlet = -3
    domainparas.xoutlet = 4
    domainparas.pitch = 2
    domainparas.blade_yshift = 0.1
    domain2d = CascadeDomain2D()
    domain2d.generate_from_cascade_parameters(domainparas)
    domain2d.plot_domain()
