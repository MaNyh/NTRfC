def test_naca(tmpdir):
    from ntrfc.turbo.airfoil_generators.naca_airfoil_creator import naca
    import numpy as np
    import pyvista as pv
    res = 480
    xs,ys = naca("6510",res,half_cosine_spacing=False,finite_te=False)


    xs,ys = naca("23112",res,half_cosine_spacing=False,finite_te=False)
