def test_cascade_meshing_gmsh():
    import tempfile
    import pyvista as pv
    import numpy as np
    from ntrfc.gmsh.turbo_cascade import generate_turbocascade
    from ntrfc.turbo.airfoil_generators.naca_airfoil_creator import naca
    from ntrfc.cascade_case.utils.domain_utils import DomainParameters
    from ntrfc.cascade_case.domain import CascadeDomain2D
    from ntrfc.filehandling.mesh import load_mesh

    ptsx, ptsy = naca("6510", 200, True)
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


    generate_turbocascade(domain2d.profilepoints, domain2d.le_index, domain2d.te_index, domain2d.yperiodic_high,
                          domain2d.yperiodic_low, domain2d.inlet, domain2d.outlet, meshpath, verbose=False)

    mesh = load_mesh(meshpath)

    assert mesh.number_of_cells > 0, "somethings wrong"
