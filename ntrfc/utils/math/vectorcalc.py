# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 19:01:50 2020

@author: malte
"""

import numpy as np
from scipy.stats import special_ortho_group
from scipy.spatial import distance
from scipy.spatial.distance import squareform, pdist

import sys
import math as m


def closest_node_index(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    return closest_index


def distant_node_index(node, nodes):
    closest_index = distance.cdist([node], nodes).argmax()
    return closest_index


def calc_largedistant_idx(x_koords, y_koords):
    """
    tested method to find indices of coordinates of a 2d-pointcloud with the biggest distance
    :param x_koords: array of xcords
    :param y_koords: array of ycords
    :return: index_1,index_2 (int)
    """
    A = np.dstack((x_koords, y_koords))[0]
    D = squareform(pdist(A))
    #    N = np.max(D)
    I = np.argmax(D)
    I_row, I_col = np.unravel_index(I, D.shape)

    index_1 = I_row
    index_2 = I_col

    return index_1, index_2


def symToMatrix(symTensor):
    """
    tested translates symmetric tensor notation to complete matrix
    :param symTensor:
    :return:
    """
    # xx,xy,xz,yy,yz,zz
    Matrix = np.array([[symTensor[0], symTensor[1], symTensor[2]],
                       [symTensor[1], symTensor[3], symTensor[4]],
                       [symTensor[2], symTensor[4], symTensor[5]]])
    return Matrix


def gradToRad(angle):
    """
    tested method to translate from grad to rad
    :param angle:
    :return:
    """
    return (angle / 180) * np.pi


def Rx(xAngle):
    """
    using radiant
    :param xAngle: angle in rad
    :return: rotation matrix
    """
    return np.array([[1, 0, 0],
                     [0, np.cos(xAngle), np.sin(xAngle)],
                     [0, -np.sin(xAngle), np.cos(xAngle)]])


def Ry(yAngle):
    """
    using radiant
    :param yAngle: angle in rad
    :return: rotation matrix
    """
    return np.array([[np.cos(yAngle), 0, np.sin(yAngle)],
                     [0, 1, 0],
                     [np.sin(yAngle), 0, np.cos(yAngle)]])


def Rz(zAngle):
    """
    using radiant
    :param zAngle: angle in rad
    :return: rotation matrix
    """
    return np.array([[np.cos(zAngle), np.sin(zAngle), 0],
                     [-np.sin(zAngle), np.cos(zAngle), 0],
                     [0, 0, 1]])


def RotFromTwoVecs(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """

    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def vecAbs(vec):
    """
    tested method to calculate the absolute value of a vector
    :param vec:
    :return:
    """
    return np.sqrt(sum(vec ** 2))


def vecDir(vec):
    """
    tested method to compute the direction of a vector
    :param vec:
    :return: unit vec
    """
    return vec / vecAbs(vec)


def posVec(vec):
    return (vec ** 2) ** .5


def findNearest(array, point):
    array = np.asarray(array)
    idx = np.array([vecAbs(i) for i in (np.abs(array - point))]).argmin()
    return idx


def eulersFromRPG(R):
    tol = sys.float_info.epsilon * 10

    if abs(R.item(0, 0)) < tol and abs(R.item(1, 0)) < tol:
        eul1 = 0
        eul2 = m.atan2(-R.item(2, 0), R.item(0, 0))
        eul3 = m.atan2(-R.item(1, 2), R.item(1, 1))
    else:
        eul1 = m.atan2(R.item(1, 0), R.item(0, 0))
        sp = m.sin(eul1)
        cp = m.cos(eul1)
        eul2 = m.atan2(-R.item(2, 0), cp * R.item(0, 0) + sp * R.item(1, 0))
        eul3 = m.atan2(sp * R.item(0, 2) - cp * R.item(1, 2), cp * R.item(1, 1) - sp * R.item(0, 1))

    """
    print("z - phi =", eul1)
    print("y - theta  =", eul2)
    print("x - psi =", eul3)

    print("checkRPGRepo")
    print(R)

    print(np.dot(np.dot(Rz(eul1),Ry(eul2)),Rx(eul3)))
    """
    return eul1, eul2, eul3


def randomUnitVec():
    """
    tested method to generate a random unit vector
    :return:
    """
    phi = np.random.uniform(0, np.pi * 2)
    costheta = np.random.uniform(-1, 1)

    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])


def randomOrthMat():
    num_dim = 3
    x = special_ortho_group.rvs(num_dim)
    return x


def ellipsoidVol(sig):
    """
    tested method to compute the ellipsoid volume by the sigma-parameters of a parametric ellipsoid
    :param sig:
    :return:
    """
    return 4 / 3 * np.pi * sig[0] * sig[1] * sig[2]


def vecProjection(direction, vector):
    unitDir = vecDir(direction)
    return np.dot(vector, unitDir) * unitDir


def vecAngle(vec1, vec2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
            angle_between((1, 0, 0), (1, 0, 0))
                0.0
            angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
    """
    absVec1 = vecAbs(vec1)
    absVec2 = vecAbs(vec2)
    return np.arccos(np.dot(vec1, vec2) / (absVec1 * absVec2))


def lineseg_dist(p, a, b):
    """
    :param p: point
    :param a: line point a
    :param b: line point b
    :return: distance
    """
    # normalized tangent vector
    d = np.divide(b - a, np.linalg.norm(b - a))

    # signed parallel distance components
    s = np.dot(a - p, d)
    t = np.dot(p - b, d)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, 0])

    # perpendicular distance component
    c = np.cross(p - a, d)

    return np.hypot(h, np.linalg.norm(c))


def line_intersection(point_a1, point_a2,
                      point_b1, point_b2):
    def det_2d(a, b):
        return a[0] * b[1] - a[1] * b[0]

    xdiff = (point_a1[0] - point_a2[0], point_b1[0] - point_b2[0])
    ydiff = (point_a1[1] - point_a2[1], point_b1[1] - point_b2[1])

    div = det_2d(xdiff, ydiff)
    if div == 0:
        return None

    d = (det_2d(point_a1, point_a2), det_2d(point_b1, point_b2))
    x = det_2d(d, xdiff) / div
    y = det_2d(d, ydiff) / div
    return x, y
