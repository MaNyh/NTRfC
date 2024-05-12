import numpy as np


def test_total_pressure_from_mach_number():
    from ntrfc.fluid.isentropic import total_pressure_from_mach_number
    # Test with some sample inputs
    kappa = 1.4
    ma = 2
    p = 100000.0

    # Calculate expected result
    expected = 782444.9066867264

    # Calculate actual result
    actual = total_pressure_from_mach_number(kappa, ma, p)

    # Check if actual result matches expected result
    assert np.isclose(expected, actual)


def test_static_pressure_from_mach_number():
    from ntrfc.fluid.isentropic import static_pressure_from_mach_number
    # Test using standard values for air at sea level
    kappa = 1.4
    mach_number = 0.8
    total_pressure = 101325.0  # Pa
    expected_static_pressure = 66471.39048022314  # Pa
    calculated_static_pressure = static_pressure_from_mach_number(kappa, mach_number, total_pressure)
    assert np.isclose(calculated_static_pressure, expected_static_pressure)


def test_total_temperature_from_mach_number():
    from ntrfc.fluid.isentropic import total_temperature_from_mach_number
    assert np.isclose(total_temperature_from_mach_number(1.4, 0, 300), 300)
    assert np.isclose(total_temperature_from_mach_number(1.4, 1.0, 300), 360.0)
    assert np.isclose(total_temperature_from_mach_number(1.4, 1.5, 300), 435)


def test_static_temperature_from_mach_number():
    from ntrfc.fluid.isentropic import static_temperature_from_mach_number
    assert np.isclose(static_temperature_from_mach_number(1.4, 0, 340.952), 340.952)
    assert np.isclose(static_temperature_from_mach_number(1.4, 1.0, 438.298), 365.24833333333333)
    assert np.isclose(static_temperature_from_mach_number(1.4, 1.5, 578.947), 399.27379310344827)


def test_local_isentropic_mach_number():
    from ntrfc.fluid.isentropic import local_isentropic_mach_number
    kappa = 1.4
    p = 100000.0
    pt = 189292.91587378542
    expected_mach = 1
    mach = local_isentropic_mach_number(kappa, p, pt)
    assert np.isclose(mach, expected_mach)


def test_calculate_isentropic_mach_number():
    from ntrfc.fluid.isentropic import calculate_isentropic_mach_number
    # Standard conditions
    isentropic_pressure = 101325  # Pa
    kappa = 1.4
    static_pressure = 101325  # Pa
    mach_number = 0.5
    gas_constant = 287.058  # J/(kg*K)
    static_temperature = 288.15  # K

    expected_output = 0.0014693046301270448

    # Call function and check output
    assert np.isclose(
        calculate_isentropic_mach_number(isentropic_pressure, kappa, static_pressure, mach_number, gas_constant,
                                         static_temperature), expected_output)
