def test_symmetric_airfoil_nostagger():
    import pyvista as pv
    import numpy as np
    from ntrfc.turbo.cascade_case.utils.domain_utils import Blade2D
    from ntrfc.turbo.airfoil_generators.naca_airfoil_creator import naca

    xs, ys = naca("0009", 480, finite_te=False, half_cosine_spacing=False)
    points = pv.PolyData(np.stack([xs, ys, np.zeros(len(xs))]).T)
    blade = Blade2D(points)
    blade.compute_all_frompoints()
    bladepts = blade.sortedpointsrolled_pv.points

    xs, ys = bladepts[:, 0], bladepts[:, 1]
    ite = blade.ite

    ite_test = len(xs) // 2
    ite_tol = 2

    assert np.abs(ite - ite_test) <= ite_tol


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
    bladepts = blade.sortedpointsrolled_pv.points

    xs, ys = bladepts[:, 0], bladepts[:, 1]
    ite = blade.ite

    ite_test = len(xs) // 2
    ite_tol = 2

    assert np.abs(ite - ite_test) <= ite_tol


def test_airfoil_nostagger():
    import pyvista as pv
    import numpy as np
    from ntrfc.turbo.cascade_case.utils.domain_utils import Blade2D
    from ntrfc.turbo.airfoil_generators.naca_airfoil_creator import naca

    xs, ys = naca("6510", 480, finite_te=False, half_cosine_spacing=False)
    points = pv.PolyData(np.stack([xs, ys, np.zeros(len(xs))]).T)
    blade = Blade2D(points)
    blade.compute_all_frompoints()
    bladepts = blade.sortedpointsrolled_pv.points

    xs, ys = bladepts[:, 0], bladepts[:, 1]
    ite = blade.ite

    ite_test = len(xs) // 2
    ite_tol = 2

    assert np.abs(ite - ite_test) <= ite_tol


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
    bladepts = blade.sortedpointsrolled_pv.points

    xs, ys = bladepts[:, 0], bladepts[:, 1]
    ite = blade.ite

    ite_test = len(xs) // 2
    ite_tol = 2

    assert np.abs(ite - ite_test) <= ite_tol


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
    blade.plot()
    bladepts = blade.sortedpointsrolled_pv.points

    xs, ys = bladepts[:, 0], bladepts[:, 1]
    ite = blade.ite

    ite_test = len(xs) // 2
    ite_tol = 2

    assert np.abs(ite - ite_test) <= ite_tol
