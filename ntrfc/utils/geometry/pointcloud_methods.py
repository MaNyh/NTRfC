import numpy as np
import pyvista as pv
from scipy.spatial import Delaunay
from itertools import product

from ntrfc.utils.math.vectorcalc import calc_largedistant_idx, vecAngle
from ntrfc.utils.pyvista_utils.line import polyline_from_points, refine_spline


def calc_concavehull(x, y, alpha):
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


def mid_length(ind_1, ind_2, sorted_poly):
    """
    calc length of a midline. currently only used in the iterative computation of LE and TE index of a profile. probably
    this method is not necessary, as it is only two lines
    :param ind_1: index LE
    :param ind_2: index TE
    :param sorted_poly: pv.PolyData sorted
    :return: length
    """

    ps_poly, ss_poly = extractSidePolys(ind_1, ind_2, sorted_poly)
    mids_poly = midline_from_sides(ps_poly, ss_poly)
    return mids_poly.length


def midline_from_sides(ps_poly, ss_poly):
    x_ps, y_ps = ps_poly.points[::, 0], ps_poly.points[::, 1]
    x_ss, y_ss = ss_poly.points[::, 0], ss_poly.points[::, 1]
    z = ps_poly.points[0][2]
    midsres = 100
    if x_ps[0] > x_ps[-1]:
        ax, ay = refine_spline(x_ps[::-1], y_ps[::-1], midsres)
    else:
        ax, ay = refine_spline(x_ps, y_ps, midsres)
    if x_ss[0] > x_ss[-1]:
        bx, by = refine_spline(x_ss[::-1], y_ss[::-1], midsres)
    else:
        bx, by = refine_spline(x_ss, y_ss, midsres)
    xmids, ymids = ((ax + bx) / 2, (ay + by) / 2)


    midsPoly = polyline_from_points(np.stack((xmids, ymids, z*np.ones(len(ymids)))).T)
    return midsPoly


def extract_vk_hk(sorted_poly, verbose=False):
    """
    This function is calculating the leading-edge and trailing edge of a long 2d-body
    The function is not 100% reliable yet. The computation is iterative and it can take a while
    Points in origPoly and sortedPoly have to have defined points on the LE and TE, otherwise a LE or TE is not defined
    and it will be random which point will be found near the LE / TE
    :param sorted_poly: sorted via calcConcaveHull
    :param verbose: bool (True -> plots, False -> silent)
    :return: returns indexes of LE(vk) and TE(hk) from sortedPoints
    """

    def checklength(ind1, ind2, sorted_poly):
        """
        calc length of a midline. currently only used in the iterative computation of LE and TE index of a profile. probably
        this method is not necessary, as it is only two lines
        :param ind1: index LE
        :param ind2: index TE
        :param sorted_poly: pv.PolyData sorted
        :return: length
        """
        psPoly, ssPoly = extractSidePolys(ind1, ind2, sorted_poly)
        midsPoly = midline_from_sides(psPoly, ssPoly)

        return midsPoly.length
    xs, ys = sorted_poly.points[::, 0], sorted_poly.points[::, 1]
    ind_1, ind_2 = calc_largedistant_idx(xs, ys)
    allowed_shift = 1
    midLength0 = checklength(ind_1, ind_2, sorted_poly)
    nopt = sorted_poly.number_of_points

    checked_combs = {}
    found = True

    while (found):

        shifts = np.arange(-allowed_shift, allowed_shift + 1)
        ind_1_ts = (shifts + ind_1) % nopt
        ind_2_ts = (shifts + ind_2) % nopt

        combs = list(product(ind_1_ts, ind_2_ts))
        # add combs entry to check weather the index combination was checked already
        for key in combs:
            if key not in checked_combs.keys():
                checked_combs[key] = False

        midLengths = []
        for ind_1_t, ind2_t in combs:
            if checked_combs[(ind_1_t, ind2_t)] == False:
                checked_combs[(ind_1_t, ind2_t)] = True
                midLengths.append(checklength(ind_1_t, ind2_t, sorted_poly))

            else:
                midLengths.append(0)
        cids = midLengths.index(max(midLengths))

        ind_1_n, ind_2_n = combs[cids]
        midLength_new = checklength(ind_1_n, ind_2_n, sorted_poly)
        if midLength_new > midLength0:
            ind_1, ind_2 = ind_1_n, ind_2_n
            midLength0 = midLength_new
            allowed_shift += 1
            found = True
        else:
            found = False

    return ind_1, ind_2


def extractSidePolys(ind_1, ind_2, sortedPoly):
    # xs, ys = list(sortedPoly.points[::, 0]), list(sortedPoly.points[::, 1])
    indices = np.arange(0, sortedPoly.number_of_points)
    # we have to split the spline differently, depending on weather the number of points is even or not
    if len(indices)%2==0:
        if ind_2 > ind_1:
            side_one_idx = indices[ind_1:ind_2+1]
            side_two_idx = np.concatenate((indices[:ind_1+1][::-1], indices[ind_2:][::-1]))
        if ind_1 > ind_2:
            side_one_idx = indices[ind_2:ind_1+1]
            side_two_idx = np.concatenate((indices[:ind_2+1][::-1], indices[ind_1:][::-1]))
    else:
        if ind_2 > ind_1:
            side_one_idx = indices[ind_1:ind_2+1]
            side_two_idx = np.concatenate((indices[:ind_1][::-1], indices[ind_2:][::-1]))
        if ind_1 > ind_2:
            side_one_idx = indices[ind_2:ind_1+1]
            side_two_idx = np.concatenate((indices[:ind_2][::-1], indices[ind_1:][::-1]))

    side_one = extract_points_fromsortedpoly(side_one_idx, sortedPoly)
    side_two = extract_points_fromsortedpoly(side_two_idx, sortedPoly)

    side_one_spline = polyline_from_points(side_one.points)
    side_two_spline = polyline_from_points(side_two.points)

    if side_one_spline.length > side_two_spline.length:

        psPoly = side_two
        ssPoly = side_one
    else:

        psPoly = side_one
        ssPoly = side_two

    return ssPoly, psPoly


def extract_points_fromsortedpoly(sorted_indices, sorted_poly):
    side_two = pv.PolyData(sorted_poly.points[sorted_indices])  # polyblade.extract_cells(index_sort)
    for arr in sorted_poly.array_names:
        side_two[arr] = sorted_poly.point_data[arr][sorted_indices]
    return side_two


def extract_geo_paras(polyblade, alpha, verbose=False):
    """
    This function is extracting profile-data as stagger-angle, midline, psPoly, ssPoly and more from a set of points
    Be careful, you need a suitable alpha-parameter in order to get the right geometry
    The calculation of the leading-edge and trailing-edge index needs time and its not 100% reliable (yet)
    Keep in mind, to check the results!
    :param polyblade: pyvista polymesh of the blade
    :param alpha: nondimensional alpha-coefficient (calcConcaveHull)
    :param verbose: bool for plots
    :return: points, psPoly, ssPoly, ind_vk, ind_hk, midsPoly, beta_leading, beta_trailing
    """
    points = polyblade.points
    xs, ys = calc_concavehull(points[:, 0], points[:, 1], alpha)

    index_sort = [np.where(points[:, 0] == xs[i])[0][0] for i in range(len(xs)) if
                  len(np.where(points[:, 0] == xs[i])) == 1 and np.where(points[:, 0] == xs[i])[0][0] == np.where(
                      points[:, 1] == ys[i])[0][0]]

    sortedPoly = pv.PolyData(polyblade.points[index_sort])  # polyblade.extract_cells(index_sort)
    for arr in polyblade.array_names:
        if sortedPoly.number_of_points==len(polyblade[arr]):
            sortedPoly[arr]=polyblade.point_data[arr][index_sort]

    ind_hk, ind_vk = extract_vk_hk(sortedPoly)
    psPoly, ssPoly = extractSidePolys(ind_hk, ind_vk, sortedPoly)
    midsPoly = midline_from_sides(psPoly, ssPoly)

    # compute angles from 2d-midline
    xmids, ymids = midsPoly.points[::, 0], midsPoly.points[::, 1]
    vk_tangent = np.stack((xmids[0] - xmids[1], ymids[0] - ymids[1], 0)).T
    hk_tangent = np.stack((xmids[-2] - xmids[-1], ymids[-2] - ymids[-1], 0)).T
    camber = np.stack((xmids[0] - xmids[-1], ymids[0] - ymids[-1], 0)).T[::-1]
    beta_leading = vecAngle(vk_tangent, np.array([0, 1, 0])) / np.pi * 180
    beta_trailing = vecAngle(hk_tangent, np.array([0, 1, 0])) / np.pi * 180
    camber_angle = vecAngle(camber, np.array([0, 1, 0])) / np.pi * 180

    if verbose:
        p = pv.Plotter()
        p.add_mesh(points, color="orange", label="points")
        p.add_mesh(psPoly, color="green", label="psPoly")
        p.add_mesh(ssPoly, color="black", label="ssPoly")
        p.add_mesh(midsPoly, color="black", label="midsPoly")
        p.add_mesh(pv.Line((0, 0, 0), (midsPoly.length, 0, 0)))
        p.add_mesh(sortedPoly.points[ind_hk],color="red",point_size=5)
        p.add_mesh(sortedPoly.points[ind_vk],color="orange",point_size=5)
        p.add_legend()
        p.show()

    return sortedPoly, psPoly, ssPoly, ind_vk, ind_hk, midsPoly, beta_leading, beta_trailing, camber_angle


def calcMidPassageStreamLine(x_mcl, y_mcl, beta1, beta2, x_inlet, x_outlet, t):
    """
    Returns mid-passage line from sceletal-line
    Returns two lists of Points representing a curve through the passage


    Input:
    x_mcl, y_mcl = Tuple
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

    p_outlet_x = x_mcl[-1] - delta_x_hk
    p_outlet_y = y_mcl[-1] - delta_y_hk

    x_mpsl = [p_inlet_x] + list(x_mcl) + [p_outlet_x]
    y_mpsl = [p_inlet_y] + list(y_mcl) + [p_outlet_y]

    for i in range(len(x_mpsl)):
        y_mpsl[i] = y_mpsl[i] + 0.5 * t

    return refine_spline(x_mpsl, y_mpsl, 1000)
