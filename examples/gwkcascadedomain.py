import numpy as np
import pyvista as pv

from ntrfc.turbo.airfoil_generators.naca_airfoil_creator import naca
from ntrfc.turbo.cascade_case.domain import CascadeWindTunnelDomain2D
from ntrfc.turbo.cascade_case.utils.domain_utils import Blade2D, CascadeWindTunnelDomain2DParameters
from ntrfc.turbo.gmsh.windtunnel_cascade import create_mesh, MeshConfig

xs, ys = naca("6510", 256)
points = pv.PolyData(np.stack([xs, ys, np.zeros(len(xs))]).T * 0.06)

gwkconfig = CascadeWindTunnelDomain2DParameters(gamma=6,
                                                gittervor=0.03,
                                                pitch=0.06,
                                                gitternach=0.04,
                                                zulauf=0.8,
                                                tailbeta=0,
                                                nblades=5
                                                )
meshconfig = MeshConfig(lc_high=0.00028,
                        lc_low=0.0024,
                        bl_size=1.8e-6,
                        progression_le=0.01,
                        progression_te=0.01,
                        bl_thickness=0.0014,
                        )

blade = Blade2D(points)
blade.compute_all_frompoints()

blade.plot()
domain2d = CascadeWindTunnelDomain2D()
domain2d.generate_from_cascade_parameters(gwkconfig, blade)
fig = domain2d.plot_domain("here.jpg")

create_mesh(domain2d, gwkconfig, meshconfig, blade, "test.msh")
