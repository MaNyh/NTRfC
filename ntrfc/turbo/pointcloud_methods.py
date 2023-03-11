import numpy as np
import pyvista as pv

from ntrfc.geometry.alphashape import calc_concavehull, auto_concaveHull
from ntrfc.geometry.line import polyline_from_points, refine_spline
from ntrfc.math.vectorcalc import vecAngle
from ntrfc.turbo.profile_tele_extraction import extract_vk_hk


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

    midsPoly = polyline_from_points(np.stack((xmids, ymids, z * np.ones(len(ymids)))).T)
    return midsPoly


def extractSidePolys(ind_1, ind_2, sortedPoly, verbose=False):
    # xs, ys = list(sortedPoly.points[::, 0]), list(sortedPoly.points[::, 1])
    indices = np.arange(0, sortedPoly.number_of_points)

    if ind_2 > ind_1:
        side_one_idx = indices[ind_1:ind_2 + 1]
        side_two_idx = np.concatenate((indices[:ind_1 + 1][::-1], indices[ind_2:][::-1]))
    elif ind_1 > ind_2:
        side_one_idx = indices[ind_2:ind_1 + 1]
        side_two_idx = np.concatenate((indices[:ind_2 + 1][::-1], indices[ind_1:][::-1]))

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

    if verbose:
        p = pv.Plotter()
        p.add_mesh(ssPoly, color="g", label="ssPoly")
        p.add_mesh(psPoly, color="b", label="psPoly")
        p.add_mesh(sortedPoly.points[ind_1], color="w", label="ind_1")
        p.add_mesh(sortedPoly.points[ind_2], color="k", label="ind_2")
        p.add_legend()
        p.show()

    return ssPoly, psPoly


def extract_points_fromsortedpoly(sorted_indices, sorted_poly):
    """
    Extract a subset of points and associated data from a sorted polygon.

    Parameters
    ----------
    sorted_indices : numpy.ndarray
        1D array of shape `(n_points,)` containing the indices of the points
        in `sorted_poly.points` that should be extracted and used to create a
        new polygon. The indices should be sorted in the order that they
        should appear in the new polygon.
    sorted_poly : pyvista.PolyData
        Polygon represented as a `PolyData` object from the `pyvista`
        library. The `sorted_poly.points` attribute should be a NumPy array
        of shape `(n_points, 3)` containing the 3D coordinates of the polygon
        vertices, and the `sorted_poly.point_data` attribute should be a
        dictionary of NumPy arrays containing additional per-vertex data.

    Returns
    -------
    pyvista.PolyData
        A new `PolyData` object representing the subset of the original polygon
        specified by `sorted_indices`, with all per-vertex data preserved.

    Notes
    -----
    This function assumes that the input polygon is sorted in a specific way.
    Specifically, the `sorted_indices` parameter should contain the indices of
    the points in the order that they should appear in the new polygon. If the
    input polygon is not sorted in this way, the resulting polygon may not be
    valid.
    """

    side_two = pv.PolyData(sorted_poly.points[sorted_indices])
    for arr in sorted_poly.array_names:
        side_two[arr] = sorted_poly.point_data[arr][sorted_indices]
    return side_two


def extract_geo_paras(polyblade, alpha=None, verbose=False):
    """
    This function is extracting profile-data as stagger-angle, midline, psPoly, ssPoly and more from a set of points
    Be careful, you need a suitable alpha-parameter in order to get the right geometry
    The calculation of the leading-edge and trailing-edge index needs time and its not 100% reliable (yet)
    Keep in mind, to check the results!
    :param polyblade: pyvista polymesh of the blade
    :param alpha: nondimensional alpha-coefficient (calcConcaveHull)
    :param verbose: bool for plots
    :return: points, psPoly, ssPoly, ind_vk_, ind_vk, midsPoly, beta_leading, beta_trailing, alpha
    """
    points = polyblade.points

    if alpha:
        xs, ys = calc_concavehull(points[:, 0], points[:, 1], alpha)
    else:
        xs, ys, alpha = auto_concaveHull(points[:, 0], points[:, 1])

    index_sort = [np.where(points[:, 0] == xs[i])[0][0] for i in range(len(xs)) if
                  len(np.where(points[:, 0] == xs[i])) == 1 and np.where(points[:, 0] == xs[i])[0][0] == np.where(
                      points[:, 1] == ys[i])[0][0]]

    sortedPoly = pv.PolyData(polyblade.points[index_sort])  # polyblade.extract_cells(index_sort)
    for arr in polyblade.array_names:
        if sortedPoly.number_of_points == len(polyblade[arr]):
            sortedPoly[arr] = polyblade.point_data[arr][index_sort]

    ind_vk, ind_hk = extract_vk_hk(sortedPoly)
    psPoly, ssPoly = extractSidePolys(ind_hk, ind_vk, sortedPoly)
    midsPoly = midline_from_sides(psPoly, ssPoly)

    # compute angles from 2d-midline
    xmids, ymids = midsPoly.points[::, 0], midsPoly.points[::, 1]
    vk_tangent = np.stack((xmids[0] - xmids[1], ymids[0] - ymids[1], 0)).T
    hk_tangent = np.stack((xmids[-2] - xmids[-1], ymids[-2] - ymids[-1], 0)).T
    chord = psPoly.points[0] - psPoly.points[-1]
    beta_leading = vecAngle(vk_tangent, -np.array([1, 0, 0])) / np.pi * 180
    beta_trailing = vecAngle(hk_tangent, -np.array([1, 0, 0])) / np.pi * 180
    camber_angle = vecAngle(chord, -np.array([1, 0, 0])) / np.pi * 180

    if verbose:
        p = pv.Plotter()
        p.add_mesh(points, color="orange", label="points")
        p.add_mesh(psPoly, color="green", label="psPoly")
        p.add_mesh(ssPoly, color="black", label="ssPoly")
        p.add_mesh(midsPoly, color="black", label="midsPoly")
        p.add_mesh(pv.Line((0, 0, 0), (midsPoly.length, 0, 0)))
        p.add_mesh(sortedPoly.points[ind_hk], color="red", point_size=5)
        p.add_mesh(sortedPoly.points[ind_vk], color="orange", point_size=5)
        p.add_legend()
        p.show()

    return sortedPoly, psPoly, ssPoly, ind_vk, ind_hk, midsPoly, beta_leading, beta_trailing, camber_angle, alpha


def calcMidPassageStreamLine(x_mcl, y_mcl, beta1, beta2, x_inlet, x_outlet, t, verbose=False):
    """
    Calculate the midpoint stream line curve through a passage.

    Parameters:
    -----------
    x_mcl : array_like
        The x-coordinates of the mid-chord line.
    y_mcl : array_like
        The y-coordinates of the mid-chord line.
    beta1 : float
        The angle in degrees representing the inflow angle at the inlet.
    beta2 : float
        The angle in degrees representing the outflow angle at the outlet.
    x_inlet : float
        The x-coordinate of the inlet position.
    x_outlet : float
        The x-coordinate of the outlet position.
    t : float
        The pitch of the midpoint stream line.
    verbose : bool, optional
        If True, a plot of the midpoint stream line is displayed.

    Returns:
    --------
    x_mpsl_ref : array_like
        The refined x-coordinates of the midpoint stream line.
    y_mpsl_ref : array_like
        The refined y-coordinates of the midpoint stream line.
    """

    delta_x_vk = x_mcl[0] - x_inlet
    delta_y_vk = np.tan(np.deg2rad(beta1)) * delta_x_vk

    p_inlet_x = x_mcl[0] - delta_x_vk
    p_inlet_y = y_mcl[0] - delta_y_vk

    delta_x_hk = x_outlet - x_mcl[-1]
    delta_y_hk = delta_x_hk * np.tan(np.deg2rad(-beta2))

    p_outlet_x = x_mcl[-1] + delta_x_hk
    p_outlet_y = y_mcl[-1] + delta_y_hk

    x_mpsl = [p_inlet_x] + list(x_mcl) + [p_outlet_x]
    y_mpsl = [p_inlet_y] + list(y_mcl) + [p_outlet_y]

    for i in range(len(x_mpsl)):
        y_mpsl[i] = y_mpsl[i] + 0.5 * t

    x_mpsl_ref, y_mpsl_ref = refine_spline(x_mpsl, y_mpsl, 1000)

    midspoly = pv.PolyData(np.stack([x_mpsl_ref, y_mpsl_ref, np.zeros(len(x_mpsl_ref))]).T)
    if verbose:
        p = pv.Plotter()
        p.add_mesh(midspoly, color="orange", point_size=5)
        p.show()
    return x_mpsl_ref, y_mpsl_ref
