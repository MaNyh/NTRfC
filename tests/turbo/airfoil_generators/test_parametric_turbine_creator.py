import numpy as np

from ntrfc.turbo.airfoil_generators.parametric_turbine_generator import create_turbine_profile


def test_create_turbine_profile():
    radius = 5.5
    axial_chord = 1.102
    tangential_chord = 0.591
    unguided_turning = 6.5 * np.pi / 180
    inlet_blade = 35 * np.pi / 180
    inlet_half_wedge = 9 * np.pi / 180
    le_r = 0.0310
    outlet_blade = -57 * np.pi / 180
    te_r = 0.0160
    n_blades = 51
    throat = 0.337
    straight_te = False

    xs, ys = create_turbine_profile(radius, axial_chord, tangential_chord,
                                    unguided_turning, inlet_blade,
                                    inlet_half_wedge,
                                    le_r, outlet_blade, te_r, n_blades, throat,
                                    straight_te)

    assert len(xs) == len(ys)
