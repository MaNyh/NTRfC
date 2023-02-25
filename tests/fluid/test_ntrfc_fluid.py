import numpy as np

from ntrfc.fluid.fluid import total_pressure, mach_number, sutherland_viscosity, ma_is, isentropic_reynolds_number, \
    isentropic_total_temperature, speed_of_sound


def test_total_pressure():
    # Test input where kappa = 1.4, mach_number = 0.5, pressure = 100
    expected_output = 118.62
    output = total_pressure(1.4, 0.5, 100)
    assert abs(output - expected_output) < 1e-2


def test_mach_number():
    # Test input where c = 340, kappa = 1.4, R_L = 287, T = 273
    expected_output = 1.02
    output = mach_number(340, 1.4, 287, 273)
    assert abs(output - expected_output) < 1e-2


def test_sutherland_viscosity():
    temperature = 300  # degrees Kelvin
    expected_dynamic_viscosity = 1.846e-05  # square meters per second

    # Call the Sutherland_Law function
    dynamic_viscosity = sutherland_viscosity(temperature)

    # Check if the output is close to the expected output
    assert np.isclose(dynamic_viscosity, expected_dynamic_viscosity, rtol=1e-10, atol=1e-10)


def test_ma_is():
    # Test input where outflow_static_pressure = 100, isentropic_exponent = 1.4, pressure = 50, velocity = 20,
    # gas_constant = 287, temperature = 300
    expected_output = 1.71
    output = ma_is(10000, 1.4, 50000, 20, 287, 300)
    assert abs(output - expected_output) < 1e-2


def test_isentropic_reynolds_number():
    # Test input where kappa = 1.4, specific_gas_constant = 287, chord_length = 1, sutherland_reference_viscosity = 1.46e-5,
    # mach_number = 0.65, pressure = 50, temperature = 300, sutherland_reference_temperature = 110.4
    expected_output = 708.95
    output = isentropic_reynolds_number(1.4, 287, 1, 1.46e-5, 0.65, 50, 300, 110.4)
    assert abs(output - expected_output) < 1e-2


def test_isentropic_total_temperature():
    # Test input where kappa = 1.4, mach_number = 0.5, temperature = 300
    expected_output = 315
    output = isentropic_total_temperature(1.4, 0.5, 300)
    assert abs(output - expected_output) < 1e-2


def test_speed_of_sound():
    assert abs(speed_of_sound(gamma=1.4, p=101325, rho=1.225) - 340.29) < 0.01
    assert abs(speed_of_sound(gamma=1.3, p=101325, rho=1.225) - 327.92) < 0.01
    assert abs(speed_of_sound(gamma=1.4, p=80000, rho=1.225) - 302.37) < 0.01
    assert abs(speed_of_sound(gamma=1.3, p=80000, rho=1.225) - 291.37) < 0.01
    assert abs(speed_of_sound(gamma=1.4, R=287.05, T=288.15) - 340.29) < 0.01
    assert abs(speed_of_sound(gamma=1.3, R=287.05, T=288.15) - 327.91) < 0.01
