import numpy as np


def test_total_pressure():
    from ntrfc.fluid.fluid import total_pressure
    # Test input where kappa = 1.4, mach_number = 0.5, pressure = 100
    expected_output = 189292.91587378542
    output = total_pressure(1.4, 1, 100000)
    assert np.isclose(expected_output, output, rtol=1e-10, atol=1e-10)


def test_mach_number():
    from ntrfc.fluid.fluid import mach_number
    # Test input where c = 340, kappa = 1.4, R_L = 287, T = 273
    expected_output = 1.02
    output = mach_number(340.3, 1.4, 287, 273)
    assert abs(output - expected_output) < 1e-2


def test_sutherland_viscosity():
    from ntrfc.fluid.fluid import sutherland_viscosity
    temperature = 300  # degrees Kelvin
    expected_dynamic_viscosity = 1.846e-05  # square meters per second

    # Call the Sutherland_Law function
    dynamic_viscosity = sutherland_viscosity(temperature)

    # Check if the output is close to the expected output
    assert np.isclose(dynamic_viscosity, expected_dynamic_viscosity, rtol=1e-10, atol=1e-10)


def test_isentropic_reynolds_number():
    from ntrfc.fluid.isentropic import calculate_isentropic_reynolds_number
    # Test input where kappa = 1.4, specific_gas_constant = 287, chord_length = 1, sutherland_reference_viscosity = 1.46e-5,
    # mach_number = 0.65, pressure = 50, temperature = 300, sutherland_reference_temperature = 110.4
    expected_output = 708.95
    output = calculate_isentropic_reynolds_number(1.4, 287, 1, 1.46e-5, 0.65, 50, 300, 110.4)
    assert abs(output - expected_output) < 1e-2


def test_isentropic_total_temperature():
    from ntrfc.fluid.isentropic import calculate_isentropic_total_temperature
    # Test input where kappa = 1.4, mach_number = 0.5, temperature = 300
    expected_output = 315
    output = calculate_isentropic_total_temperature(1.4, 0.5, 300)
    assert abs(output - expected_output) < 1e-2
