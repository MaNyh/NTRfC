import numpy as np
from scipy import optimize
from scipy.spatial import Delaunay
from scipy.spatial import distance
from scipy.spatial.distance import cdist


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
        for ia, ib, ic in tri.simplices:
            pa = points[ia]
            pb = points[ib]
            pc = points[ic]
            a = np.linalg.norm(pa - pb)
            b = np.linalg.norm(pb - pc)
            c = np.linalg.norm(pc - pa)
            s = (a + b + c) / 2.0
            area = np.sqrt(s * (s - a) * (s - b) * (s - c))
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

    if not len(boundary_lst) == 0:
        for i in range(len(boundary_lst[0])):
            x_new.append(points[boundary_lst[0][i][0]][0])
            y_new.append(points[boundary_lst[0][i][0]][1])

    return x_new, y_new


def calculate_minimal_distances(points):
    distances = cdist(points, points)  # Calculate pairwise distances
    np.fill_diagonal(distances, np.inf)  # Set diagonal elements to infinity
    minimal_distances = np.min(distances, axis=1)  # Find minimal distance for each point
    return minimal_distances


def auto_concaveHull(xs, ys):
    # Define the loss function
    def loss(alpha, xs, ys):
        xd, yd = calc_concavehull(xs, ys, alpha)
        if len(xd) == 0:
            return 1e10
        closed_points = np.stack([xd + [xd[0]], yd + [yd[0]]]).T
        points_orig = np.stack([xs, ys]).T
        centers = (closed_points[:-1] + closed_points[1:]) / 2
        distances = distance.cdist(centers, points_orig).min(axis=0)

        loss_distance_shape = np.max(distances)
        loss_norm_numpts = (len(xs) - len(xd)) / len(xs)

        return loss_distance_shape * 2 + loss_norm_numpts

    points = np.column_stack((xs, ys))
    # Compute the pairwise distances between all points
    distances = cdist(points, points)

    # Get the smallest and largest distances
    smallest_distance = np.min(distances[np.nonzero(distances)])  # Ignore zero distances
    largest_distance = np.max(distances)
    bounds_first_stage = [(smallest_distance, largest_distance / 2)]
    x0_first_stage = np.mean(bounds_first_stage, axis=1)

    result_first_stage = optimize.minimize(fun=loss, x0=x0_first_stage, args=(xs, ys,), method='Powell',
                                           bounds=bounds_first_stage).x[0]

    bounds_second_stage = [(result_first_stage / 2, result_first_stage)]

    result_second_stage = optimize.minimize(fun=loss, x0=result_first_stage * 0.9, args=(xs, ys,), method='Nelder-mead',
                                            bounds=bounds_second_stage).x[0]

    bounds_third_stage = [(1e-9, result_second_stage)]
    result = optimize.minimize(fun=loss, x0=result_second_stage, args=(xs, ys,), method='TNC',
                               bounds=bounds_third_stage).x[0]

    xans, yans = calc_concavehull(xs, ys, result)
    return xans, yans, result
