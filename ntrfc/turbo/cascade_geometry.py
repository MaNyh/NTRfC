import numpy as np

from ntrfc.geometry.line import refine_spline


def calcmidpassagestreamline(x_mcl, y_mcl, beta1, beta2, x_inlet, x_outlet, t):
    """
    Returns mid-passage line from sceletal-line
    Returns two lists of Points representing a curve through the passage
    Input:
    x_mcl = Tuple
    y_mcl = Tuple
    beta1, beta2 = Angle in deg - Beta = Anströmwinkel
    x_inlet, x_outlet = scalar - representing position x-component of in/outlet
    t = scalar pitch
    """

    delta_x_vk = x_mcl[0] - x_inlet
    delta_y_vk = np.tan(np.deg2rad(beta1 - 90)) * delta_x_vk

    p_inlet_x = x_mcl[0] - delta_x_vk
    p_inlet_y = y_mcl[0] - delta_y_vk

    delta_x_hk = x_outlet - x_mcl[-1]
    delta_y_hk = delta_x_hk * np.tan(np.deg2rad(beta2 - 90))

    p_outlet_x = x_mcl[-1] + delta_x_hk
    p_outlet_y = y_mcl[-1] + delta_y_hk

    x_mpsl = [p_inlet_x] + list(x_mcl) + [p_outlet_x]
    y_mpsl = [p_inlet_y] + list(y_mcl) + [p_outlet_y]

    for i in range(len(x_mpsl)):
        y_mpsl[i] = y_mpsl[i] + 0.5 * t

    return refine_spline(x_mpsl, y_mpsl, 1000)
