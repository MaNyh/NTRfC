def test_cascade_meshing_gmshcascade(tmpdir):
    import pyvista as pv
    import numpy as np
    from ntrfc.turbo.gmsh.turbo_cascade import generate_turbocascade, MeshConfig
    from ntrfc.turbo.airfoil_generators.naca_airfoil_creator import naca
    from ntrfc.turbo.cascade_case.utils.domain_utils import CascadeDomain2DParameters, Blade2D
    from ntrfc.turbo.cascade_case.domain import CascadeDomain2D
    from ntrfc.filehandling.mesh import load_mesh
    from ntrfc.geometry.alphashape import auto_concaveHull

    ptsx, ptsy = naca("6510", 200, False)
    ptsx_n, ptsy_n, _ = auto_concaveHull(ptsx, ptsy)
    # create a 3d pointcloud using pv.PolyData, all z values are 0
    pts = pv.PolyData(np.c_[ptsx_n, ptsy_n, np.zeros(len(ptsx_n))])

    domainparams = CascadeDomain2DParameters(xinlet=-3, xoutlet=4, pitch=2, blade_yshift=0.1)
    blade = Blade2D(pts)
    blade.compute_all_frompoints(1)
    domain2d = CascadeDomain2D()
    domain2d.generate_from_cascade_parameters(domainparams, blade)

    meshpath = tmpdir / "test.cgns"

    meshconfig = MeshConfig()

    di = 0.04

    meshconfig.max_lc = di
    meshconfig.min_lc = di / 10
    meshconfig.bl_thickness = di * 1.6
    meshconfig.bl_growratio = 1.2
    meshconfig.bl_size = 1.0e-5
    meshconfig.wake_length = blade.camber_length * 1
    meshconfig.wake_width = blade.camber_length * .1
    meshconfig.wake_lc = di * 0.5
    meshconfig.fake_yShiftCylinder = 0
    meshconfig.bladeres = int((blade.ps_pv.length + blade.ss_pv.length) / (meshconfig.min_lc * 4))
    meshconfig.progression_le_halfss = 1.05
    meshconfig.progression_halfss_te = 0.95
    meshconfig.progression_te_halfps = 1.05
    meshconfig.progression_halfps_le = 0.95
    meshconfig.spansize = 0.01
    meshconfig.spanres = 1

    generate_turbocascade(domain2d,
                          meshconfig,
                          str(meshpath))

    mesh = load_mesh(meshpath)

    assert mesh.number_of_cells > 0, "somethings wrong"


def test_cascade_meshing_gmshwindtunnel(tmpdir):
    from ntrfc.turbo.cascade_case.domain import CascadeWindTunnelDomain2D
    from ntrfc.turbo.cascade_case.utils.domain_utils import Blade2D, CascadeWindTunnelDomain2DParameters
    from ntrfc.turbo.gmsh.windtunnel_cascade import create_mesh, MeshConfig
    from ntrfc.filehandling.mesh import load_mesh
    import pyvista as pv
    import numpy as np
    from ntrfc.turbo.airfoil_generators.naca_airfoil_creator import naca
    from ntrfc.geometry.alphashape import auto_concaveHull

    xs, ys = naca("6510", 256)
    xs_n,ys_n,_ = auto_concaveHull(xs,ys)
    points = pv.PolyData(np.stack([xs_n, ys_n, np.zeros(len(xs_n))]).T * 0.06)

    gwkconfig = CascadeWindTunnelDomain2DParameters(gamma=6,
                                                    gittervor=0.03,
                                                    pitch=0.06,
                                                    gitternach=0.04,
                                                    zulauf=0.8,
                                                    tailbeta=0,
                                                    nblades=5
                                                    )

    meshconfig = MeshConfig(lc_high=0.0008,
                            lc_low=0.008,
                            bl_size=10e-6,
                            progression_le=0.01,
                            progression_te=0.01,
                            bl_thickness=0.0014,
                            )
    blade = Blade2D(points)
    blade.compute_all_frompoints()

    domain2d = CascadeWindTunnelDomain2D()
    domain2d.generate_from_cascade_parameters(gwkconfig, blade)
    domain2d.plot_domain(tmpdir / "test.png")
    meshpath = tmpdir / "test.cgns"
    create_mesh(domain2d, gwkconfig, meshconfig, blade, str(meshpath))
    mesh = load_mesh(meshpath)
    assert mesh.number_of_cells > 0, "somethings wrong"
