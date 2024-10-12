def test_symmetric_airfoil_nostagger():
    import pyvista as pv
    import numpy as np
    from ntrfc.turbo.cascade_case.utils.domain_utils import Blade2D
    from ntrfc.turbo.airfoil_generators.naca_airfoil_creator import naca

    xs, ys = naca("0009", 480, finite_te=False, half_cosine_spacing=False)
    points = pv.PolyData(np.stack([xs, ys, np.zeros(len(xs))]).T)
    blade = Blade2D(points)
    blade.compute_all_frompoints()

    ite = blade.ite

    ite_test = 507
    ite_tol = 2

    ile = blade.ile
    ile_test = 27
    ile_tol = 1

    assert np.abs(ite - ite_test) <= ite_tol
    assert np.abs(ile - ile_test) <= ile_tol

def test_symmetric_airfoil_stagger():
    import pyvista as pv
    import numpy as np
    from ntrfc.turbo.cascade_case.utils.domain_utils import Blade2D
    from ntrfc.turbo.airfoil_generators.naca_airfoil_creator import naca

    xs, ys = naca("0009", 480, finite_te=False, half_cosine_spacing=False)
    points = pv.PolyData(np.stack([xs, ys, np.zeros(len(xs))]).T)
    points = pv.DataSet.rotate_z(points, angle=20, inplace=True)

    blade = Blade2D(points)
    blade.compute_all_frompoints()

    ite = blade.ite
    ite_test = 140
    ite_tol = 1

    ile = blade.ile
    ile_test = 620
    ile_tol = 1

    assert np.abs(ite - ite_test) <= ite_tol
    assert np.abs(ile - ile_test) <= ile_tol


def test_airfoil_nostagger():
    import pyvista as pv
    import numpy as np
    from ntrfc.turbo.cascade_case.utils.domain_utils import Blade2D
    from ntrfc.turbo.airfoil_generators.naca_airfoil_creator import naca

    xs, ys = naca("6510", 480, finite_te=False, half_cosine_spacing=False)
    points = pv.PolyData(np.stack([xs, ys, np.zeros(len(xs))]).T)
    blade = Blade2D(points)
    blade.compute_all_frompoints()

    ite = blade.ite
    ite_test = 146
    ite_tol = 1

    ile = blade.ile
    ile_test = 626
    ile_tol = 1

    assert np.abs(ite - ite_test) <= ite_tol
    assert np.abs(ile - ile_test) <= ile_tol

def test_airfoil_stagger():
    import pyvista as pv
    import numpy as np
    from ntrfc.turbo.cascade_case.utils.domain_utils import Blade2D
    from ntrfc.turbo.airfoil_generators.naca_airfoil_creator import naca

    xs, ys = naca("6510", 480, finite_te=False, half_cosine_spacing=False)
    points = pv.PolyData(np.stack([xs, ys, np.zeros(len(xs))]).T)
    points = pv.DataSet.rotate_z(points, angle=20, inplace=True)

    blade = Blade2D(points)
    blade.compute_all_frompoints()
    ite = blade.ite
    ite_test = 141
    ite_tol = 1

    ile = blade.ile
    ile_test=621
    ile_tol = 1

    assert np.abs(ite - ite_test) <= ite_tol
    assert np.abs(ile - ile_test) <= ile_tol


def test_t106():
    import pyvista as pv
    import os
    import importlib
    import numpy as np
    from ntrfc.turbo.cascade_case.utils.domain_utils import Blade2D

    # we need a display some situations like a cicd run
    if os.getenv('DISPLAY') is None:
        pv.start_xvfb()  # Start X virtual framebuffer (Xvfb)

    profilepoints_file = importlib.resources.files("ntrfc") / "data/turbine_cascade/profilepoints.txt"
    points = np.loadtxt(profilepoints_file)

    blade = Blade2D(points)
    blade.compute_all_frompoints()

    ite = blade.ite

    ite_test = 38
    ite_tol = 1

    ile = blade.ile
    ile_test = 87
    ile_tol = 1

    assert np.abs(ite - ite_test) <= ite_tol
    assert np.abs(ile - ile_test) <= ile_tol
