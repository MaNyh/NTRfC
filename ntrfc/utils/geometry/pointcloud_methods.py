import numpy as np
import pyvista as pv
from scipy.spatial import Delaunay
from scipy.spatial.distance import squareform, pdist
from itertools import product


from ntrfc.utils.pyvista_utils.line import polyline_from_points, refine_spline

def calcConcaveHull(x, y, alpha):
    """
    origin: https://stackoverflow.com/questions/50549128/boundary-enclosing-a-given-set-of-points/50714300#50714300
    """
    points = []
    for i in range(len(x)):
        points.append([x[i], y[i]])

    points = np.asarray(points)

    def alpha_shape(points, alpha, only_outer=True):
        """
        Compute the alpha shape (concave hull) of a set of points.
        :param points: np.array of shape (n,2) points.
        :param alpha: alpha value.
        :param only_outer: boolean value to specify if we keep only the outer border
        or also inner edges.
        :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
        the indices in the points array.
        """

        assert points.shape[0] > 3, "Need at least four points"

        def add_edge(edges, i, j):
            """
            Add an edge between the i-th and j-th points,
            if not in the list already
            """
            if (i, j) in edges or (j, i) in edges:
                # already added
                assert (j, i) in edges, "Can't go twice over same directed edge right?"
                if only_outer:
                    # if both neighboring triangles are in shape, it's not a boundary edge
                    edges.remove((j, i))
                return
            edges.add((i, j))

        tri = Delaunay(points)
        edges = set()
        # Loop over triangles:
        # ia, ib, ic = indices of corner points of the triangle
        for ia, ib, ic in tri.vertices:
            pa = points[ia]
            pb = points[ib]
            pc = points[ic]
            # Computing radius of triangle circumcircle
            # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
            a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
            b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
            c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
            s = (a + b + c) / 2.0

            A = (s * (s - a) * (s - b) * (s - c))
            if A > 0:
                area = np.sqrt(A)

                circum_r = a * b * c / (4.0 * area)
                if circum_r < alpha:
                    add_edge(edges, ia, ib)
                    add_edge(edges, ib, ic)
                    add_edge(edges, ic, ia)
        return edges

    def find_edges_with(i, edge_set):
        i_first = [j for (x, j) in edge_set if x == i]
        i_second = [j for (j, x) in edge_set if x == i]
        return i_first, i_second

    def stitch_boundaries(edges):
        edge_set = edges.copy()
        boundary_lst = []
        while len(edge_set) > 0:
            boundary = []
            edge0 = edge_set.pop()
            boundary.append(edge0)
            last_edge = edge0
            while len(edge_set) > 0:
                i, j = last_edge
                j_first, j_second = find_edges_with(j, edge_set)
                if j_first:
                    edge_set.remove((j, j_first[0]))
                    edge_with_j = (j, j_first[0])
                    boundary.append(edge_with_j)
                    last_edge = edge_with_j
                elif j_second:
                    edge_set.remove((j_second[0], j))
                    edge_with_j = (j, j_second[0])  # flip edge rep
                    boundary.append(edge_with_j)
                    last_edge = edge_with_j

                if edge0[0] == last_edge[1]:
                    break

            boundary_lst.append(boundary)
        return boundary_lst

    edges = alpha_shape(points, alpha)
    boundary_lst = stitch_boundaries(edges)
    x_new = []
    y_new = []

    for i in range(len(boundary_lst[0])):
        x_new.append(points[boundary_lst[0][i][0]][0])
        y_new.append(points[boundary_lst[0][i][0]][1])

    return x_new, y_new


def midLength(ind_1, ind_2, sortedPoly):
    """
    calc length of a midline
    :param ind_1: index LE
    :param ind_2: index TE
    :param sortedPoly: pv.PolyData sorted
    :return: length
    """
    psPoly, ssPoly = extractSidePolys(ind_1, ind_2, sortedPoly)
    midsPoly = midline_from_sides(ind_1, ind_2, sortedPoly.points, psPoly, ssPoly)
    arclength = midsPoly.compute_arc_length()["arc_length"]
    midslength = sum(arclength)
    return midslength

def midline_from_sides(ind_hk, ind_vk, points, psPoly, ssPoly):
    x_ps, y_ps = psPoly.points[::, 0], psPoly.points[::, 1]
    x_ss, y_ss = ssPoly.points[::, 0], ssPoly.points[::, 1]

    midsres = 100
    if x_ps[0] < x_ps[-1]:
        ax, ay = refine_spline(x_ps[::-1], y_ps[::-1], midsres)
    else:
        ax, ay = refine_spline(x_ps, y_ps, midsres)
    if x_ss[0] < x_ss[-1]:
        bx, by = refine_spline(x_ss[::-1], y_ss[::-1], midsres)
    else:
        bx, by = refine_spline(x_ss, y_ss, midsres)
    xmids, ymids = ((ax + bx) / 2, (ay + by) / 2)
    xmids = np.array(xmids)[::-1][1:-1]
    ymids = np.array(ymids)[::-1][1:-1]
    xmids[0] = points[ind_vk][0]
    ymids[0] = points[ind_vk][1]
    xmids[-1] = points[ind_hk][0]
    ymids[-1] = points[ind_hk][1]
    midsPoly = polyline_from_points(np.stack((xmids, ymids, np.zeros(len(ymids)))).T)
    return midsPoly

def calc_largedistant_idx(x_koords, y_koords):
    A = np.dstack((x_koords, y_koords))[0]
    D = squareform(pdist(A))
    #    N = np.max(D)
    I = np.argmax(D)
    I_row, I_col = np.unravel_index(I, D.shape)

    index_1 = I_row
    index_2 = I_col

    return index_1, index_2


def extract_vk_hk(sortedPoly, verbose=False):
    """
    This function is calculating the leading-edge and trailing edge of a long 2d-body
    The function is not 100% reliable yet. The computation is iterative and it can take a while
    Points in origPoly and sortedPoly have to have defined points on the LE and TE, otherwise a LE or TE is not defined
    and it will be random which point will be found near the LE / TE
    :param origPoly: all original points, unsorted
    :param sortedPoly: sorted via calcConcaveHull
    :param verbose: bool (True -> plots, False -> silent)
    :return: returns indexes of LE(vk) and TE(hk) from sortedPoints
    """

    xs, ys = sortedPoly.points[::, 0], sortedPoly.points[::, 1]
    ind_1, ind_2 = calc_largedistant_idx(xs, ys)
    allowed_shift = 1
    midLength0 = midLength(ind_1, ind_2, sortedPoly)
    nopt = sortedPoly.number_of_points

    checked_combs = {}
    found = True
    while (found):

        shifts = np.arange(-allowed_shift, allowed_shift + 1)
        ind_1_ts = (shifts + ind_1) % nopt
        ind_2_ts = (shifts + ind_2) % nopt

        combs = list(product(ind_1_ts, ind_2_ts))
        for key in combs:
            if key not in checked_combs.keys():
                checked_combs[key] = False

        midLengths = []
        for ind_1_t, ind2_t in combs:
            if checked_combs[(ind_1_t, ind2_t)] == False:
                checked_combs[(ind_1_t, ind2_t)] = True
                midLengths.append(midLength(ind_1_t, ind2_t, sortedPoly))
            else:
                midLengths.append(0)
        cids = midLengths.index(max(midLengths))

        ind_1_n, ind_2_n = combs[cids]
        midLength_new = midLength(ind_1_n, ind_2_n, sortedPoly)
        if midLength_new > midLength0:
            ind_1, ind_2 = ind_1_n, ind_2_n
            midLength0 = midLength_new
            allowed_shift += 1
            found = True
        else:
            found = False

    if sortedPoly.points[ind_1][0] > sortedPoly.points[ind_2][0]:
        ind_vk = ind_2
        ind_hk = ind_1
    else:
        ind_vk = ind_1
        ind_hk = ind_2
    return ind_hk, ind_vk


def extractSidePolys(ind_hk, ind_vk, sortedPoly):
    xs, ys = list(sortedPoly.points[::, 0]), list(sortedPoly.points[::, 1])

    if ind_vk < ind_hk:
        x_ss = xs[ind_vk:ind_hk + 1]
        y_ss = ys[ind_vk:ind_hk + 1]

        y_ps = ys[ind_hk:] + ys[:ind_vk + 1]
        x_ps = xs[ind_hk:] + xs[:ind_vk + 1]

    else:
        x_ss = xs[ind_hk:ind_vk + 1]
        y_ss = ys[ind_hk:ind_vk + 1]

        y_ps = ys[ind_vk:] + ys[:ind_hk + 1]
        x_ps = xs[ind_vk:] + xs[:ind_hk + 1]

    psl_helper = polyline_from_points(np.stack((x_ps, y_ps, np.zeros(len(x_ps)))).T)
    ssl_helper = polyline_from_points(np.stack((x_ss, y_ss, np.zeros(len(x_ss)))).T)

    if psl_helper.length > ssl_helper.length:

        psPoly = pv.PolyData(ssl_helper.points)
        ssPoly = pv.PolyData(psl_helper.points)
    else:

        psPoly = pv.PolyData(psl_helper.points)
        ssPoly = pv.PolyData(ssl_helper.points)


    return ssPoly, psPoly
