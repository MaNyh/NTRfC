import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import KDTree
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, LineString

from ntrfc.geometry.line import refine_spline
from ntrfc.geometry.plane import inside_poly
from ntrfc.math.vectorcalc import findNearest, vecDir


def clean_sites(sites, boundary, tolerance_factor=5e-2):
    # Build the KDTree from the "boundary" point cloud.
    tree = KDTree(boundary)

    # Query the KDTree to find the nearest neighbors for each point in "point_cloud_a."

    distances, _ = tree.query(sites)
    deg_res = 4

    theta = np.linspace(0, 2 * np.pi, 360 * deg_res)
    cleaned = []
    radii = []
    for xc, yc, r in zip(sites[:, 0], sites[:, 1], distances):

        circlepoints = np.stack([np.cos(theta) * r + xc, np.sin(theta) * r + yc]).T
        circle_dists, _ = tree.query(circlepoints)

        half_circle_idx = 180 * deg_res
        circle_dists_180_deg = circle_dists[:half_circle_idx] + circle_dists[half_circle_idx:]
        # min_dist = np.argmin(circle_dists)
        atol = r * tolerance_factor
        circle_in_skeleton = np.where(np.isclose(circle_dists_180_deg, 0, atol=atol))[0]

        if len(circle_in_skeleton) > 0:
            min_dist_at = np.argmin(circle_dists[circle_in_skeleton])
            if np.isclose(circle_dists[circle_in_skeleton[min_dist_at]], 0, atol=atol / 2):
                cleaned.append([xc, yc])
                radii.append(r)
    print(f"ratio of cleaned sites: {len(cleaned) / len(sites)}")

    return np.array(cleaned), np.array(radii)


def extract_vk_hk(sortedPoly):
    points = sortedPoly.points

    diag_dist = np.sqrt((sortedPoly.bounds[1] - sortedPoly.bounds[0]) ** 2 + (
        sortedPoly.bounds[3] - sortedPoly.bounds[2]) ** 2)
    points_2d_closed_refined = pointcloud_to_profile(points)

    sites_raw_clean, radii = voronoi_skeleton_sites(points_2d_closed_refined)

    le_ind, te_ind = skeletonline_completion(diag_dist, points, points_2d_closed_refined, sites_raw_clean)

    return le_ind, te_ind


def skeletonline_completion(diag_dist, points, points_2d_closed_refined, sites_raw_clean):
    sites_refined_clean_tuple = refine_spline(sites_raw_clean[:, 0], sites_raw_clean[:, 1], 100)
    sites_refined_clean = np.stack([sites_refined_clean_tuple[0], sites_refined_clean_tuple[1]]).T
    shapelypoly = Polygon(points_2d_closed_refined)
    shapelymidline = LineString(sites_refined_clean)
    # i need to extend thhe shapelymidline to the boundary of the polygon
    # Get the coordinates of the start and end points
    start_coords = np.array(shapelymidline.coords[0])
    end_coords = np.array(shapelymidline.coords[-1])
    # Compute the direction vector
    direction_start = diag_dist * vecDir(-(shapelymidline.coords[1] - start_coords))
    direction_end = diag_dist * vecDir(-(shapelymidline.coords[-2] - end_coords))
    # Extend the line by 1 unit in both directions
    extended_start = LineString([start_coords, start_coords + direction_start])
    extended_end = LineString([end_coords, end_coords + direction_end])
    # Compute the intersection with the polygon
    intersection_start = extended_start.intersection(shapelypoly)
    intersection_end = extended_end.intersection(shapelypoly)
    intersection_point_start = np.array(intersection_start.coords)[1]
    intersection_point_end = np.array(intersection_end.coords)[1]
    # find closest point index of points and intersections
    le_ind = findNearest(points[:, :2], intersection_point_start)
    te_ind = findNearest(points[:, :2], intersection_point_end)
    return le_ind, te_ind


def voronoi_skeleton_sites(points_2d_closed_refined):
    vor = Voronoi(points_2d_closed_refined)
    voronoi_sites_inside = vor.vertices[inside_poly(points_2d_closed_refined, vor.vertices)]

    sort_indices = np.argsort(voronoi_sites_inside[:, 0])
    sites_inside_sorted = voronoi_sites_inside[sort_indices]

    clean_sites_inside, radii = clean_sites(sites_inside_sorted, points_2d_closed_refined)
    return clean_sites_inside, radii


def pointcloud_to_profile(points):
    points_2d_closed = np.vstack((points[:, :2], points[:, :2][0]))
    tck, u = splprep(points_2d_closed.T, u=None, s=0.0, per=1, k=3)
    res = 10000
    u_new = np.linspace(u.min(), u.max(), res)
    x_new, y_new = splev(u_new, tck, der=0)
    points_2d_closed_refined = np.stack([x_new, y_new]).T

    return points_2d_closed_refined
