import numpy as np
import pyvista as pv
from scipy.spatial import Voronoi

from ntrfc.geometry.line import lines_from_points, polyline_from_points, refine_spline
from ntrfc.geometry.plane import inside_poly
from ntrfc.math.vectorcalc import findNearest, vecDir


def get_centroids(triangles):
    centroids = []
    for triangle in triangles:
        x = (triangle[0][0] + triangle[1][0] + triangle[2][0]) / 3
        y = (triangle[0][1] + triangle[1][1] + triangle[2][1]) / 3
        centroids.append((x, y))
    return centroids


def extract_vk_hk(sortedPoly, verbose=False):
    points = sortedPoly.points
    points_2d = points[:, :2]

    # https://link.springer.com/article/10.1007/s13272-013-0088-6
    # Use the Voronoi algorithm to divide the space into Voronoi cells

    vor = Voronoi(points_2d)

    # Identify the center points of the Voronoi cells (also known as Voronoi sites)
    voronoi_sites = vor.vertices
    voronoi_sites_inside = voronoi_sites[inside_poly(points_2d, voronoi_sites)]
    vor_sites_x = voronoi_sites_inside[::, 0]
    vor_sites_y = voronoi_sites_inside[::, 1]
    vor_sites_sorted = sorted(zip(vor_sites_x, vor_sites_y))
    vor_sites_x_sorted, vor_sites_y_sorted = zip(*vor_sites_sorted)
    cama_evenx, cama_eveny = refine_spline(vor_sites_x_sorted, vor_sites_y_sorted, 100)
    camberpoints = np.stack([cama_evenx, cama_eveny, np.zeros(len(cama_eveny))]).T[1:-1]
    camberline = lines_from_points(camberpoints)
    LE_camber = camberline.extract_cells(0)
    LE_dir = vecDir(LE_camber.points[-1] - LE_camber.points[0])
    TE_camber = camberline.extract_cells(camberline.number_of_cells - 1)
    TE_dir = vecDir(TE_camber.points[0] - TE_camber.points[-1])
    X = points[::, 0]
    Y = points[::, 1]
    Z = X * 0
    profilepoly = polyline_from_points(np.vstack([np.stack([X, Y, Z]).T, np.stack([X[0], Y[0], Z[0]]).T]))

    camber_le_extension = pv.Line(LE_camber.points[0] - LE_dir * 10, camberpoints[0], resolution=10000)
    camber_te_extension = pv.Line(camberpoints[-1], TE_camber.points[0] - TE_dir * 10, resolution=10000)
    camberline_extended = lines_from_points(np.vstack([camber_le_extension.points,
                                                       camberpoints[1:-2],
                                                       camber_te_extension.points]))

    helpersurface = profilepoly.copy().extrude([0, 0, -1], inplace=True)
    helpersurface = helpersurface.translate([0, 0, .5], inplace=True)

    camberline_computed = camberline_extended.clip_surface(helpersurface, invert=False)

    le_ind = findNearest(np.stack([X, Y, Z]).T, camberline_computed.points[0])
    te_ind = findNearest(np.stack([X, Y, Z]).T, camberline_computed.points[-1])

    # Return the camber line points
    return le_ind, te_ind
