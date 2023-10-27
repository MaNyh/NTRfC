def test_cascade_meshing_gmsh(tmpdir):
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
    domainparams.xinlet = -1.4
    domainparams.xoutlet = 3
    domainparams.pitch = 1.6
    domainparams.blade_yshift = 0.05
    domain2d = CascadeDomain2D()
    domain2d.generate_from_cascade_parameters(domainparams)

    meshpath = tmpdir / "test.cgns"

    meshconfig = MeshConfig()

    di = 0.04

    meshconfig.max_lc = di
    meshconfig.min_lc = di / 10
    meshconfig.bl_thickness = di * 1.6
    meshconfig.bl_growratio = 1.2
    meshconfig.bl_size = 1.0e-5
    meshconfig.wake_length = domain2d.chordlength * 1
    meshconfig.wake_width = domain2d.chordlength * .1
    meshconfig.wake_lc = di * 0.5
    meshconfig.fake_yShiftCylinder = 0
    meshconfig.bladeres = int((domainparams.pspoly.length + domainparams.sspoly.length) / (meshconfig.min_lc * 4))
    meshconfig.progression_le_halfss = 1.05
    meshconfig.progression_halfss_te = 0.95
    meshconfig.progression_te_halfps = 1.05
    meshconfig.progression_halfps_le = 0.95
    meshconfig.spansize = 0.01
    meshconfig.spanres = 1

    generate_turbocascade(domain2d,
                          meshconfig,
                          str(meshpath), verbose=False)

    mesh = load_mesh(meshpath)

    assert mesh.number_of_cells > 0, "somethings wrong"
