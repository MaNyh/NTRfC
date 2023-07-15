def test_cascade_meshing_gmsh():
    import tempfile
    import pyvista as pv
    import numpy as np
    from ntrfc.gmsh.turbo_cascade import generate_turbocascade, MeshConfig
    from ntrfc.turbo.airfoil_generators.naca_airfoil_creator import naca
    from ntrfc.cascade_case.utils.domain_utils import DomainParameters
    from ntrfc.cascade_case.domain import CascadeDomain2D
    from ntrfc.filehandling.mesh import load_mesh

    ptsx, ptsy = naca("6510", 200, False)
    # create a 3d pointcloud using pv.PolyData, all z values are 0
    pts = pv.PolyData(np.c_[ptsx, ptsy, np.zeros(len(ptsx))])
    domainparams = DomainParameters()
    domainparams.generate_params_by_pointcloud(pts)
    domainparams.xinlet = -2
    domainparams.xoutlet = 3
    domainparams.pitch = 1
    domainparams.blade_yshift = 0.05
    domain2d = CascadeDomain2D()
    domain2d.generate_from_cascade_parameters(domainparams)

    meshpath = tempfile.mkdtemp() + "/test.cgns"

    meshconfig = MeshConfig()
    meshconfig.max_lc = 0.04
    meshconfig.min_lc = 0.01
    meshconfig.bl_thickness = 0.05
    meshconfig.bl_growratio = 1.2
    meshconfig.bl_size = 1.7e-5
    meshconfig.wake_length = 1
    meshconfig.wake_width = 0.1
    meshconfig.wake_lc = 0.01
    meshconfig.fake_yShiftCylinder = 0

    generate_turbocascade(domain2d,
                          meshconfig,
                          meshpath, verbose=False)

    mesh = load_mesh(meshpath)

    assert mesh.number_of_cells > 0, "somethings wrong"
