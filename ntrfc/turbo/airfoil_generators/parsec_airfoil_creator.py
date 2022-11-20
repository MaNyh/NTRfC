# Generate and plot the contour of an airfoil
# using the PARSEC parameterization

# Repository & documentation:
# http://github.com/dqsis/parsec-airfoils
# -------------------------------------


# Import libraries
from __future__ import division

from math import sqrt, tan, pi

import numpy as np


# User function pcoef
def pcoef(xte, yte, rle, x_cre, y_cre, d2ydx2_cre, th_cre, surface):
    """
    evaluate the PARSEC coefficients
    """

    # Initialize coefficients
    coef = np.zeros(6)

    # 1st coefficient depends on surface (pressure or suction)
    if surface.startswith('p'):
        coef[0] = -sqrt(2 * rle)
    else:
        coef[0] = sqrt(2 * rle)

    # Form system of equations
    A = np.array([
        [xte ** 1.5, xte ** 2.5, xte ** 3.5, xte ** 4.5, xte ** 5.5],
        [x_cre ** 1.5, x_cre ** 2.5, x_cre ** 3.5, x_cre ** 4.5,
         x_cre ** 5.5],
        [1.5 * sqrt(xte), 2.5 * xte ** 1.5, 3.5 * xte ** 2.5,
         4.5 * xte ** 3.5, 5.5 * xte ** 4.5],
        [1.5 * sqrt(x_cre), 2.5 * x_cre ** 1.5, 3.5 * x_cre ** 2.5,
         4.5 * x_cre ** 3.5, 5.5 * x_cre ** 4.5],
        [0.75 * (1 / sqrt(x_cre)), 3.75 * sqrt(x_cre), 8.75 * x_cre ** 1.5,
         15.75 * x_cre ** 2.5, 24.75 * x_cre ** 3.5]
    ])

    B = np.array([
        [yte - coef[0] * sqrt(xte)],
        [y_cre - coef[0] * sqrt(x_cre)],
        [tan(th_cre * pi / 180) - 0.5 * coef[0] * (1 / sqrt(xte))],
        [-0.5 * coef[0] * (1 / sqrt(x_cre))],
        [d2ydx2_cre + 0.25 * coef[0] * x_cre ** (-1.5)]
    ])

    # Solve system of linear equations
    X = np.linalg.solve(A, B)

    # Gather all coefficients
    coef[1:6] = X[0:5, 0]

    # Return coefficients
    return coef


def parsec_airfoil_gen(pparray, halfsinespacing=True, resolution=2000):
    # TE & LE of airfoil (normalized, chord = 1)
    xle = 0.0
    # yle = 0.0
    xte = 1.0
    yte = 0.0

    # LE radius
    rle = pparray[0]

    # Pressure (lower) surface parameters
    x_pre = pparray[1]
    y_pre = pparray[2]
    d2ydx2_pre = pparray[3]
    th_pre = pparray[4]

    # Suction (upper) surface parameters
    x_suc = pparray[5]
    y_suc = pparray[6]
    d2ydx2_suc = pparray[7]
    th_suc = pparray[8]

    # Evaluate pressure (lower) surface coefficients
    cf_pre = pcoef(xte, yte, rle,
                   x_pre, y_pre, d2ydx2_pre, th_pre,
                   'pre')

    # Evaluate suction (upper) surface coefficients
    cf_suc = pcoef(xte, yte, rle,
                   x_suc, y_suc, d2ydx2_suc, th_suc,
                   'suc')

    # Evaluate pressure (lower) surface points

    if halfsinespacing:
        beta = np.linspace(0.0, pi, resolution)
        halfsinespacing = [(0.5 * (1.0 - np.cos(x))) for x in beta]
        xx_pre = np.array(halfsinespacing[::-1])
        # xx_suc = np.array(halfsinespacing)

    else:  # Evaluate pressure (lower) surface points
        xx_pre = np.linspace(xte, xle, 101)
        # Evaluate suction (upper) surface points
        # xx_suc = np.linspace(xle, xte, 101)

    yy_pre = (cf_pre[0] * xx_pre ** (1 / 2) +
              cf_pre[1] * xx_pre ** (3 / 2) +
              cf_pre[2] * xx_pre ** (5 / 2) +
              cf_pre[3] * xx_pre ** (7 / 2) +
              cf_pre[4] * xx_pre ** (9 / 2) +
              cf_pre[5] * xx_pre ** (11 / 2)
              )

    # Evaluate suction (upper) surface points
    # xx_suc = x * np.linspace(xle, xte, res)
    xx_suc = np.array(halfsinespacing)
    yy_suc = (cf_suc[0] * xx_suc ** (1 / 2) +
              cf_suc[1] * xx_suc ** (3 / 2) +
              cf_suc[2] * xx_suc ** (5 / 2) +
              cf_suc[3] * xx_suc ** (7 / 2) +
              cf_suc[4] * xx_suc ** (9 / 2) +
              cf_suc[5] * xx_suc ** (11 / 2)
              )

    ps_points = np.stack([xx_pre, yy_pre, np.zeros(len(xx_pre))]).T
    ss_points = np.stack([xx_suc, yy_suc, np.zeros(len(xx_suc))]).T
    return np.concatenate((ps_points, ss_points))
