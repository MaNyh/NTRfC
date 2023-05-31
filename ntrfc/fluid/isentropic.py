import numpy as np

from ntrfc.fluid.fluid import mach_number, total_pressure


def p_t_is_from_mach(kappa, mach_number, static_pressure):
    # Calculates total pressure in isentropic flow
    # Source: https://www.grc.nasa.gov/www/BGH/isentrop.html
    total_pressure = static_pressure * pow(1.0 + (kappa - 1.0) / 2.0 * pow(mach_number, 2.0), (kappa / (kappa - 1.0)))
    return total_pressure


def p_is_from_mach(kappa, ma, p_t_is):
    # Calculates static pressure in isentropic flow
    # Source: https://www.grc.nasa.gov/www/BGH/isentrop.html
    p_is = p_t_is / pow(1.0 + (kappa - 1.0) / 2.0 * pow(ma, 2.0), (kappa / (kappa - 1.0)))
    return p_is


def temp_t_is(kappa, ma, T):
    # Calculates total temperature in isentropic flow
    # Source: https://www.grc.nasa.gov/www/BGH/isentrop.html
    T_t_is = T / (((1.0 + (kappa - 1.0) * 0.5 * ma ** 2.0)) ** (-1.0))
    return T_t_is


def temp_is(kappa, ma, Tt):
    # Calculates static temperature in isentropic flow
    # Source: https://www.grc.nasa.gov/www/BGH/isentrop.html
    T = Tt / (1 + ((kappa - 1) / 2.0) * ma ** 2)
    return T


def mach_is_x(kappa, p, pt):
    # Calculates local isentropic Mach number
    if p > 0 and pt > 0 and pt >= p:
        y = np.sqrt(2 / (kappa - 1) * ((pt / p) ** ((kappa - 1) / kappa) - 1))
    else:
        y = 0.0
    return y


def isentropic_mach_number(isentropic_pressure, kappa, static_pressure, mach, gas_constant, static_temperature):
    """
    Calculates the isentropic Mach number.

    Parameters:
        isentropic_pressure (float): Isentropic pressure of the flow.
        kappa (float): Specific heat ratio of the gas.
        static_pressure (float): Static pressure of the flow.
        mach (float): Mach number of the flow.
        gas_constant (float): Gas constant of the gas.
        static_temperature (float): Static temperature of the flow.

    Returns:
        float: Isentropic Mach number.

    """
    # Calculate the total pressure
    total_pressure = p_t_is_from_mach(kappa, mach_number(mach, kappa, gas_constant, static_temperature),
                                      static_pressure)

    # Calculate the dynamic pressure
    dynamic_pressure = total_pressure - isentropic_pressure

    # Calculate the isentropic Mach number
    isentropic_mach_number = np.sqrt(
        2.0 / (kappa - 1.0) * (pow(1.0 + (dynamic_pressure / isentropic_pressure), (kappa - 1.0) / kappa) - 1.0))

    return isentropic_mach_number


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
