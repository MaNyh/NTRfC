import numpy as np


def sutherland_viscosity(temperature, sutherland_constant=1.458e-06, reference_temperature=110.4):
    """
    Calculate the dynamic viscosity of a gas using Sutherland's Law.

    Parameters:
    - temperature (float): the temperature of the gas in degrees Kelvin
    - sutherland_constant (float): the Sutherland's constant for the gas (default value is for air)
    - reference_temperature (float): the reference temperature for the gas, in degrees Kelvin (default value is for air)

    Returns:
    - dynamic_viscosity (float): the dynamic viscosity of the gas, in units of square meters per second
    """

    dynamic_viscosity = sutherland_constant * (temperature) ** (0.5) / (1 + reference_temperature / temperature)
    return dynamic_viscosity


def total_pressure(kappa, mach_number, pressure):
    """
    Calculates the total pressure at a point in a gas using the isentropic flow equations.
    https://www.grc.nasa.gov/www/BGH/isentrop.html

    Parameters:
    - kappa: the ratio of specific heats for the gas at the point
    - mach_number: the Mach number at the point
    - pressure: the pressure at the point

    Returns:
    - the total pressure at the point
    """
    total_pressure = pressure * (1 + (kappa - 1) / 2 * mach_number ** 2) ** (kappa / (kappa - 1))
    return total_pressure


def mach_number(speed_of_sound, specific_heat_ratio, specific_gas_constant, temperature):
    """
    Calculates the Mach number at a point in a gas.

    Parameters:
    - speed_of_sound: the speed of sound in the gas
    - specific_heat_ratio: the ratio of specific heats for the gas
    - specific_gas_constant: the specific gas constant for the gas
    - temperature: the temperature of the gas

    Returns:
    - the Mach number at the point
    """
    mach_number = speed_of_sound / ((specific_heat_ratio * specific_gas_constant * temperature) ** 0.5)
    return mach_number


def ma_is(outflow_static_pressure, isentropic_exponent, pressure, velocity, gas_constant, temperature):
    """
    Calculate the isentropic Mach number at the outflow of a system.

    Parameters:
        outflow_static_pressure (float): static pressure at the outflow (Pa)
        isentropic_exponent (float): isentropic exponent of the gas
        pressure (float): pressure at a reference point in the flow (Pa)
        density (float): density at the reference point (kg/m^3)
        velocity (float): velocity at the reference point (m/s)
        gas_constant (float): gas constant of the gas (J/kg*K)
        temperature (float): temperature at the reference point (K)

    Returns:
        float: isentropic Mach number at the outflow
    """
    reference_point_total_pressure = total_pressure(isentropic_exponent,
                                                    mach_number(velocity, isentropic_exponent, gas_constant,
                                                                temperature), pressure)
    q2th = reference_point_total_pressure - outflow_static_pressure
    isentropic_mach_number = np.sqrt(2.0 / (isentropic_exponent - 1.0) * (
        pow(1.0 + (q2th / outflow_static_pressure), (isentropic_exponent - 1.0) / isentropic_exponent) - 1.0))
    return isentropic_mach_number


def isentropic_reynolds_number(kappa, specific_gas_constant, chord_length, sutherland_reference_viscosity,
                               mach_number, pressure, temperature,
                               sutherland_reference_temperature):
    """
    Calculates the isentropic Reynolds number at a point in a gas flow.

    Parameters:
    - kappa: the ratio of specific heats for the gas
    - specific_gas_constant: the specific gas constant for the gas
    - chord_length: the chord length of the body or structure
    - sutherland_reference_viscosity: the Sutherland reference viscosity
    - mach_number: the Mach number at the point
    - pressure: the pressure at the reference point
    - temperature: the temperature at the reference point
    - velocity_magnitude: the velocity magnitude at the reference point
    - isobaric_heat_constant: the isobaric heat constant
    - sutherland_reference_temperature: the Sutherland reference temperature

    Returns:
    - the isentropic Reynolds number at the point
    """
    total_temperature = isentropic_total_temperature(kappa, mach_number, temperature)
    iso_temperature = total_temperature / (1 + (kappa - 1) / 2 * mach_number ** 2)
    y = (kappa / specific_gas_constant) ** 0.5 * chord_length / sutherland_reference_viscosity * (
        mach_number * pressure * (iso_temperature + sutherland_reference_temperature) / iso_temperature ** 2)
    return y


def isentropic_total_temperature(kappa, mach_number, temperature):
    """
    Calculates the isentropic total temperature at a point in a gas.

    https://www.grc.nasa.gov/www/BGH/isentrop.html
    Eq #7

    Parameters:
    - kappa: the ratio of specific heats for the gas
    - mach_number: the Mach number at the point
    - temperature: the temperature at the point

    Returns:
    - the isentropic total temperature at the point
    """
    isentropic_total_temperature = temperature / (1 + (kappa - 1) / 2 * mach_number ** 2) ** -1
    return isentropic_total_temperature


def speed_of_sound(*, gamma, R=None, p=None, rho=None, T=None):
    """Calculate the speed of sound based on the given equation.
    https://www.grc.nasa.gov/www/BGH/isentrop.html

    Args:
        gamma (float): The specific heat ratio.
        p (float): The pressure of the gas.
        rho (float): The density of the gas.
        R (float): The specific gas constant.
        T (float): The temperature of the gas.

    Returns:
        float: The speed of sound.
    """
    if not T and not R:
        a = np.sqrt(gamma * p / rho)
    elif not rho:
        a = np.sqrt(gamma * R * T)
    return a
