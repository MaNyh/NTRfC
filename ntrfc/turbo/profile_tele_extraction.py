import numpy as np
import pyvista as pv
from scipy.spatial import Voronoi
from scipy.interpolate import splprep, splev
from scipy.spatial import KDTree
from ntrfc.geometry.line import lines_from_points, polyline_from_points
from ntrfc.geometry.plane import inside_poly
from ntrfc.math.vectorcalc import findNearest, vecDir, compute_minmax_distance_in_pointcloud, randomUnitVec


def clean_sites(sites, boundary, tolerance_factor =3e-2):

    # Build the KDTree from the "boundary" point cloud.
    tree = KDTree(boundary)

    # Query the KDTree to find the nearest neighbors for each point in "point_cloud_a."

    distances, _ = tree.query(sites)
    deg_res = 2

    theta = np.linspace(0, 2 * np.pi, 360*deg_res)
    cleaned = []
    radii = []
    for xc, yc, r in zip(sites[:, 0], sites[:, 1], distances):

        circlepoints = np.stack([ np.cos(theta) * r+xc,  np.sin(theta) * r+yc]).T
        circle_dists, _ = tree.query(circlepoints)

        half_circle_idx = 180*deg_res
        circle_dists_180_deg = circle_dists[:half_circle_idx]+circle_dists[half_circle_idx:]
        # min_dist = np.argmin(circle_dists)
        atol = r*tolerance_factor
        circle_in_skeleton = np.where(np.isclose(circle_dists_180_deg,0,atol=atol))[0]

        if len(circle_in_skeleton) > 0:
            min_dist_at = np.argmin(circle_dists[circle_in_skeleton])
            if np.isclose(circle_dists[circle_in_skeleton[min_dist_at]],0,atol=atol/2):
                cleaned.append([xc, yc])
                radii.append(r)
    print(f"ratio of cleaned sites: {len(cleaned)/len(sites)}")

    return np.array(cleaned), np.array(radii)



def extract_vk_hk(sortedPoly, verbose=False):
    points = sortedPoly.points

    points_2d_closed_refined = pointcloud_to_profile(points)

    sites_raw_clean, radii= voronoi_skeleton_sites(points_2d_closed_refined)

    tck, u = splprep(sites_raw_clean.T, u=None, s=0.000005, per=0, k=3)
    res = 4000
    u_new = np.linspace(u.min(), u.max(), res)
    x_center, y_center = splev(u_new, tck, der=0)
    le_ind, te_ind = skeletonline_completion(x_center, y_center, points)

    return le_ind, te_ind


def voronoi_skeleton_sites(points_2d_closed_refined):
    vor = Voronoi(points_2d_closed_refined)
    voronoi_sites_inside = vor.vertices[inside_poly(points_2d_closed_refined, vor.vertices)]

    sort_indices = np.argsort(voronoi_sites_inside[:, 0])
    sites_inside_sorted = voronoi_sites_inside[sort_indices]

    clean_sites_inside, radii = clean_sites(sites_inside_sorted, points_2d_closed_refined)
    return clean_sites_inside, radii



def skeletonline_completion(x_center, y_center, points):
    camberpoints = np.stack([x_center, y_center, np.zeros(len(y_center))]).T
    camberline = lines_from_points(camberpoints)
    LE_camber = camberline.extract_cells(0)
    LE_dir = vecDir(LE_camber.points[-1] - LE_camber.points[0])
    TE_camber = camberline.extract_cells(camberline.number_of_cells - 1)
    TE_dir = vecDir(TE_camber.points[0] - TE_camber.points[-1])
    X = points[::, 0]
    Y = points[::, 1]
    Z = X * 0
    profilepoly = polyline_from_points(np.vstack([np.stack([X, Y, Z]).T, np.stack([X[0], Y[0], Z[0]]).T]))
    chord = compute_minmax_distance_in_pointcloud(points, minmax="max")
    camber_le_extension = pv.Line(LE_camber.points[0] - LE_dir * chord / 2, camberpoints[0], resolution=400)
    camber_te_extension = pv.Line(camberpoints[-1], TE_camber.points[0] - TE_dir * chord / 2, resolution=400)
    camberline_extended = lines_from_points(np.vstack([camber_le_extension.points,
                                                       camberpoints[1:-2],
                                                       camber_te_extension.points]))
    helpersurface = profilepoly.copy().extrude([0, 0, -1], inplace=True)
    helpersurface = helpersurface.translate([0, 0, .5], inplace=True)
    camberline_computed = camberline_extended.clip_surface(helpersurface, invert=False)
    le_ind = findNearest(np.stack([X, Y, Z]).T, camberline_computed.points[0])
    te_ind = findNearest(np.stack([X, Y, Z]).T, camberline_computed.points[-1])



    return le_ind, te_ind


def pointcloud_to_profile(points):
    points_2d_closed = np.vstack((points[:, :2], points[:, :2][0]))
    tck, u = splprep(points_2d_closed.T, u=None, s=0.0, per=1, k=3)
    res = 10000
    u_new = np.linspace(u.min(), u.max(), res)
    x_new, y_new = splev(u_new, tck, der=0)
    points_2d_closed_refined = np.stack([x_new, y_new]).T
    return points_2d_closed_refined
