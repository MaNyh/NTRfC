def calc_inflow_cp(px, pt1, p1):
    """
    :param px: pressure at position
    :param pt1: total pressure inlet
    :param p1: pressure inlet
    :return: lift coefficient
    """
    cp = (px - p1) / (pt1 - p1)
    return cp


# Totaldruckverlustbeiwert
def calc_zeta(pt1, pt2x, p2):
    """
    Calculates the Total Pressure Loss Coefficient (Zeta) for a fluid system.

    Parameters:
    -----------
    pt1 : float : Upstream total pressure [Pa]
    pt2x : float : Downstream total pressure [Pa]
    p2 : float : Downstream static pressure [Pa]

    Returns:
    --------
    zeta : float : Total Pressure Loss Coefficient (Zeta) [dimensionless]
    """
    zeta = (pt1 - pt2x) / (pt1 - p2)
    return zeta


def calc_cf(tau_w, u_inf, rho_inf):
    """
    Calculates skin friction coefficient

    :param tau_w: wall shear stress
    :type tau_w: float
    :param u_inf: freestream velocity
    :type u_inf: float
    :param rho_inf: freestream density
    :type rho_inf: float
    :return: skin friction coefficient
    :rtype: float
    """
    cf = 2 * tau_w / (rho_inf * u_inf ** 2)
    return cf
