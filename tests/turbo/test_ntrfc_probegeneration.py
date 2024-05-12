import numpy as np
import pyvista as pv


def test_createprofileprobes():
    from ntrfc.turbo.probegeneration import create_profileprobes
    from ntrfc.turbo.airfoil_generators.naca_airfoil_creator import naca
    from ntrfc.turbo.cascade_case.utils.domain_utils import Blade2D
    naca_code = "6009"
    angle = 10  # deg
    res = 480
    xs, ys = naca(naca_code, res, half_cosine_spacing=False)
    sorted_poly = pv.PolyData(np.stack([xs, ys, np.zeros(len(xs))]).T)
    sorted_poly.rotate_z(angle, inplace=True)
    blade = Blade2D(sorted_poly)
    blade.compute_all_frompoints()
    n_psprobes = 24
    n_ssprobes = 36
    probes_ss, probes_ps = create_profileprobes(blade.ss_pv, blade.ps_pv, 1, n_ssprobes, n_psprobes, tolerance=1e-6)
    assert probes_ps.number_of_points == n_psprobes, "number of pressure side probes not correct"
    assert probes_ss.number_of_points == n_ssprobes, "number of suction side probes not correct"


def test_create_midpassageprobes():
    from ntrfc.turbo.probegeneration import create_midpassageprobes
    from ntrfc.turbo.airfoil_generators.naca_airfoil_creator import naca
    from ntrfc.turbo.cascade_case.utils.domain_utils import Blade2D
    naca_code = "6009"
    angle = 10  # deg
    res = 480
    xs, ys = naca(naca_code, res, half_cosine_spacing=False)
    sorted_poly = pv.PolyData(np.stack([xs, ys, np.zeros(len(xs))]).T)
    sorted_poly.rotate_z(angle, inplace=True)
    blade = Blade2D(sorted_poly)
    blade.compute_all_frompoints()

    nop = 40
    midspan_probes = create_midpassageprobes(1, -0.3, 0.3, 0.1, blade.beta_le, blade.beta_te, blade.skeletonline_pv,
                                             nop)
    assert midspan_probes.number_of_points == nop, "number of probes on midpassage line not correct"


def test_stagnationpointprobes():
    from ntrfc.turbo.probegeneration import create_stagnationpointprobes
    from ntrfc.turbo.airfoil_generators.naca_airfoil_creator import naca

    naca_code = "6509"
    angle = 30  # deg
    res = 420
    xs, ys = naca(naca_code, res, half_cosine_spacing=False)
    sorted_poly = pv.PolyData(np.stack([xs[:-1], ys[:-1], np.zeros(len(xs) - 1)]).T)
    sorted_poly.rotate_z(angle, inplace=True)
    probes = create_stagnationpointprobes(1, 20, sorted_poly, 0, np.array([1, 0, 0]), 1)
