import numpy as np

from ntrfc.fluid.fluid import mach_number


def total_pressure_from_mach_number(kappa, mach_number, static_pressure):
    # Calculates total pressure in isentropic flow
    # Source: https://www.grc.nasa.gov/www/BGH/isentrop.html
    total_pressure = static_pressure * pow(1.0 + (kappa - 1.0) / 2.0 * pow(mach_number, 2.0), (kappa / (kappa - 1.0)))
    return total_pressure


def static_pressure_from_mach_number(kappa, ma, p_t_is):
    # Calculates static pressure in isentropic flow
    # Source: https://www.grc.nasa.gov/www/BGH/isentrop.html
    p_is = p_t_is / pow(1.0 + (kappa - 1.0) / 2.0 * pow(ma, 2.0), (kappa / (kappa - 1.0)))
    return p_is


def total_temperature_from_mach_number(kappa, ma, T):
    # Calculates total temperature in isentropic flow
    # Source: https://www.grc.nasa.gov/www/BGH/isentrop.html
    T_t_is = T / (((1.0 + (kappa - 1.0) * 0.5 * ma ** 2.0)) ** (-1.0))
    return T_t_is


def static_temperature_from_mach_number(kappa, ma, Tt):
    # Calculates static temperature in isentropic flow
    # Source: https://www.grc.nasa.gov/www/BGH/isentrop.html
    T = Tt / (1 + ((kappa - 1) / 2.0) * ma ** 2)
    return T


def local_isentropic_mach_number(kappa, p_blade, p_frestream):
    # Calculates local isentropic Mach number
    y = np.sqrt(2 / (kappa - 1) * ((p_frestream / p_blade) ** ((kappa - 1) / kappa) - 1))

    return y


def calculate_isentropic_mach_number(isentropic_pressure, kappa, static_pressure, mach, gas_constant,
                                     static_temperature):
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
    total_pressure = total_pressure_from_mach_number(kappa, mach_number(mach, kappa, gas_constant, static_temperature),
                                                     static_pressure)

    # Calculate the dynamic pressure
    dynamic_pressure = total_pressure - isentropic_pressure

    # Calculate the isentropic Mach number
    isentropic_mach_number = np.sqrt(
        2.0 / (kappa - 1.0) * (pow(1.0 + (dynamic_pressure / isentropic_pressure), (kappa - 1.0) / kappa) - 1.0))

    return isentropic_mach_number


def calculate_isentropic_reynolds_number(kappa, specific_gas_constant, chord_length, sutherland_reference_viscosity,
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
    total_temperature = calculate_isentropic_total_temperature(kappa, mach_number, temperature)
    iso_temperature = total_temperature / (1 + (kappa - 1) / 2 * mach_number ** 2)
    y = (kappa / specific_gas_constant) ** 0.5 * chord_length / sutherland_reference_viscosity * (
        mach_number * pressure * (iso_temperature + sutherland_reference_temperature) / iso_temperature ** 2)
    return y


def calculate_isentropic_total_temperature(kappa, mach_number, temperature):
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
