import numpy as np
import open3d as o3d
import pyvista as pv
import shapely
from scipy.interpolate import splev
from scipy.interpolate import splprep
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, LineString

from ntrfc.math.vectorcalc import findNearest, vecDir


def filter_voronoi_sites(voronoi_sites_3d,sortedPoly,refinedPoints):
    points_3d = voronoi_sites_3d

    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    # 2. Voxel downsampling
    start_size = 0.0005
    goals = 0
    goal_le = []
    goal_te = []

    upper = 150
    lower = 75
    while goals != 512:

        if start_size <=0:
            start_size = 0.00025

        pcd_new = pcd.voxel_down_sample(voxel_size=start_size)

        if len(pcd_new.points) > upper:
            start_size += 0.0005*np.random.rand()
        elif len(pcd_new.points) < lower:
            start_size -= 0.0005*np.random.rand()
        elif (lower <= len(pcd_new.points) <= upper):
            goals += 1

            filtered_sites = np.asarray(pcd_new.points)
            vr = pv.PolyData(filtered_sites)

            x_new = vr.points[:, 0]
            y_new = vr.points[:, 1]

            # sort x_new and y_new along x
            sort_indices = np.argsort(x_new)
            x_new = x_new[sort_indices]
            y_new = y_new[sort_indices]
            le_ind, te_ind, skeletonline_complete = skeletonline_completion(2, sortedPoly,
                                                                            refinedPoints,
                                                                            np.stack([x_new[1:-1], y_new[1:-1]]).T)

            goal_le.append(le_ind)
            goal_te.append(te_ind)

            start_size += (np.random.rand()-0.5)*1e-3

    #o3d.visualization.draw_geometries([pcd])   # open3d.visualization.draw_geometries([outlier_cloud])
    return np.bincount(goal_le).argmax(),np.bincount(goal_te).argmax()


def extract_vk_hk(sortedPoly: pv.PolyData) -> (int, int):
    voronoires = 32000

    points_orig = sortedPoly.points
    points_2d_closed_refined_voronoi = pointcloud_to_profile(points_orig, voronoires)

    voronoi_sites = voronoi_skeleton_sites(points_2d_closed_refined_voronoi)
    le_ind, te_ind = filter_voronoi_sites(np.stack([voronoi_sites[:, 0], voronoi_sites[:, 1], np.zeros(len(voronoi_sites))]).T,
                                          sortedPoly.points,
                                          points_2d_closed_refined_voronoi)


    return le_ind, te_ind


def skeletonline_completion(diag_dist, points, points_2d_closed_refined, sites_raw_clean):
    shapelypoly = Polygon(points_2d_closed_refined).convex_hull
    shapelymidline = LineString(sites_raw_clean)
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

    skeleton_points = np.concatenate([np.array([points[le_ind][:2]]), sites_raw_clean, np.array([points[te_ind][:2]])])
    zeros_column = np.zeros((skeleton_points.shape[0], 1))

    skeletonline_complete = pv.PolyData(np.hstack((skeleton_points, zeros_column)))

    return le_ind, te_ind, skeletonline_complete


def voronoi_skeleton_sites(points_2d_closed_refined_voronoi):
    vor = Voronoi(points_2d_closed_refined_voronoi)
    polygon = shapely.geometry.Polygon(points_2d_closed_refined_voronoi)#.convex_hull
    is_inside = [shapely.geometry.Point(i).within(polygon) for i in vor.vertices]
    voronoi_sites_inside = vor.vertices[is_inside]

    sort_indices = np.argsort(voronoi_sites_inside[:, 0])
    sites_inside_sorted = voronoi_sites_inside[sort_indices]

    return sites_inside_sorted



def pointcloud_to_profile(points, res):
    tck, u = splprep(points.T, u=None, s=0.0, per=1, k=3)
    u_new = np.linspace(u.min(), u.max(), res)
    a_new = splev(u_new, tck, der=0)
    points_2d_closed_refined = np.stack([a_new[0], a_new[1]]).T

    return points_2d_closed_refined
