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


def mach_number(speed, specific_heat_ratio, specific_gas_constant, temperature):
    """
    Calculates the Mach number at a point in a gas.

    Parameters:
    - speed: the speed of the gas
    - specific_heat_ratio: the ratio of specific heats for the gas
    - specific_gas_constant: the specific gas constant for the gas
    - temperature: the temperature of the gas

    Returns:
    - the Mach number at the point
    """
    mach_number = speed / ((specific_heat_ratio * specific_gas_constant * temperature) ** 0.5)
    return mach_number
