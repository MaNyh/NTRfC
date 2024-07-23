import cv2
import numpy as np
import pyvista as pv
import shapely
from scipy.interpolate import splev
from scipy.interpolate import splprep
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, LineString
from skimage.morphology import skeletonize
from ntrfc.geometry.line import lines_from_points
from ntrfc.turbo.pointcloud_methods import extractSidePolys, midline_from_sides
from ntrfc.math.vectorcalc import findNearest, vecDir
from scipy.optimize import differential_evolution
from dataclasses import dataclass

def detect_inliers_tukey(data):
    """
    Detect inliers using Tukey's method.

    Parameters:
    - data: 1D array-like data

    Returns:
    - Array of boolean values indicating whether each data point is an inlier
    """
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data >= lower_bound) & (data <= upper_bound)


def detect_inliers_mad(data):
    """
    Detect inliers using Median Absolute Deviation (MAD) method.

    Parameters:
    - data: 1D array-like data

    Returns:
    - Array of boolean values indicating whether each data point is an inlier
    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    threshold = 3.5 * mad  # Adjust multiplier as needed

    lower_bound = median - threshold
    upper_bound = median + threshold
    return (data >= lower_bound) & (data <= upper_bound)


def detect_inliers_zscore(data, threshold=2):
    """
    Detect outliers using Z-Score method.

    Parameters:
    - data: 1D array-like data
    - threshold: Z-Score threshold for identifying outliers (default=3)

    Returns:
    - Array of boolean values indicating whether each data point is an outlier
    """
    z_scores = np.abs((data - np.mean(data)) / np.std(data))
    return z_scores < threshold


def clean_sites(voronoi_sites, skeletonize_sites):
    # Compute pairwise distances
    distances = cdist(skeletonize_sites, voronoi_sites)

    # Find minimal distance for each point in A
    min_distances = np.min(distances, axis=1)

    inliers_turkey = detect_inliers_tukey(min_distances)
    inliers_mad = detect_inliers_mad(min_distances)
    inliers_zscore = detect_inliers_zscore(min_distances)

    inlier_indices = np.where(inliers_turkey * inliers_zscore * inliers_mad)[0]

    valid_midline = skeletonize_sites[inlier_indices]
    return valid_midline


def compute_midline_error(ind_1_start, ind_2_start, sortedPoly, shapelypoly):
    ps_pv, ss_pv = extractSidePolys(ind_1_start, ind_2_start, sortedPoly)

    mid = lines_from_points(midline_from_sides(ps_pv, ss_pv, res=100).points)

    mid_normals = mid.extrude([0, 0, 1]).compute_normals().slice("z").compute_cell_sizes()


    normals = mid_normals["Normals"]
    lengths = mid_normals["Length"]
    centers = mid_normals.cell_centers().points

    raylength=np.sum(mid_normals["Length"])

    ray_positive = centers + raylength * normals
    ray_negative = centers - raylength * normals

    # Compute the intersection with the polygon of all rays
    intersections_positive = [LineString([centers[i], ray_positive[i]]).intersection(shapelypoly) for i in
                              range(len(centers))]
    intersections_negative = [LineString([centers[i], ray_negative[i]]).intersection(shapelypoly) for i in
                              range(len(centers))]

    # compute length of each intersection
    lengths_positive = np.array([line.length for line in intersections_positive])
    lengths_negative = np.array([line.length for line in intersections_negative])

    differences = np.abs(lengths_positive - lengths_negative) / ((lengths_positive + lengths_negative) / 2)

    rms_error = np.sqrt(np.mean(differences ** 2))

    return rms_error


def extract_vk_hk(sortedPoly: pv.PolyData) -> (int, int):
    voronoires = 32000
    skeletonres = 6000

    points_orig = sortedPoly.points
    npts = len(points_orig)
    points_2d_closed_refined_voronoi = pointcloud_to_profile(points_orig, voronoires)
    points_2d_closed_refined_skeletonize = pointcloud_to_profile(points_orig, skeletonres)

    #create sheply polygon from refined pointset
    shapelypoly = Polygon(points_2d_closed_refined_voronoi)
    raylength = 1000

    voronoi_sites = voronoi_skeleton_sites(points_2d_closed_refined_voronoi)
    skeletonize_sites = skeletonize_skeleton_sites(points_2d_closed_refined_skeletonize)
    valid_midline = clean_sites(voronoi_sites, skeletonize_sites)
    sort_indices = np.argsort(valid_midline[:, 0])
    valid_midline_sorted = valid_midline[sort_indices]
    smoothing = 0.00001

    u_new = np.arange(0, 1, 1 / 1024)
    (tck, u), fp, ier, msg = splprep((valid_midline_sorted[::, 0], valid_midline_sorted[::, 1]), u=None, per=0, k=3,
                                     s=smoothing,
                                     full_output=True)  # s = optional parameter (default used here)
    x_new, y_new = splev(u_new, tck, der=0)


    le_ind, te_ind, skeletonline_complete = skeletonline_completion(2, points_orig, points_2d_closed_refined_voronoi,
                                                                    np.stack([x_new, y_new]).T)

    rms_error_ref = compute_midline_error(le_ind, te_ind,sortedPoly,shapelypoly)

    # Define the objective function
    def objective_function(indices, sortedPoly, shapelypoly):
        le_ind, te_ind = int(indices[0]), int(indices[1])
        rms_error_le_up = compute_midline_error(le_ind, te_ind, sortedPoly, shapelypoly)
        return rms_error_le_up

    @dataclass
    class SkeletonResult:
        le_ind: int
        te_ind: int
        rms_error: float


    # optimize te
    search_te_pos = 3
    te_pos_ind = te_ind
    search_te_neg = 3
    te_neg_ind = te_ind
    search_le_pos = 3
    le_pos_ind = le_ind
    search_le_neg = 3
    le_neg_ind = le_ind
    search_le_neg_te_pos = 3
    le_neg_te_pos_le_ind = le_ind
    le_neg_te_pos_te_ind = te_ind
    search_le_pos_te_neg = 3
    le_pos_te_neg_le_ind = le_ind
    le_pos_te_neg_te_ind = te_ind
    search_le_neg_te_neg = 3
    le_neg_te_neg_le_ind = le_ind
    le_neg_te_neg_te_ind = te_ind
    search_le_pos_te_pos = 3
    le_pos_te_pos_le_ind = le_ind
    le_pos_te_pos_te_ind = te_ind

    skeletonresults = []
    skeletonresults.append(SkeletonResult(le_ind, te_ind, rms_error_ref))


    while search_te_pos > 0:
        te_pos_ind = (te_pos_ind + 1) % npts
        rms_error_te_pos = objective_function([le_ind, te_pos_ind], sortedPoly, shapelypoly)
        skeletonresults.append(SkeletonResult(le_ind, te_pos_ind, rms_error_te_pos))
        if rms_error_te_pos > rms_error_ref:
            search_te_pos -= 1
    while search_te_neg > 0:
        te_neg_ind = (te_neg_ind - 1) % npts
        rms_error_te_neg = objective_function([le_ind, te_neg_ind], sortedPoly, shapelypoly)
        skeletonresults.append(SkeletonResult(le_ind, te_neg_ind, rms_error_te_neg))
        if rms_error_te_neg > rms_error_ref:
            search_te_neg -= 1
    while search_le_pos > 0:
        le_pos_ind = (le_pos_ind + 1) % npts
        rms_error_le_pos = objective_function([le_pos_ind, te_ind], sortedPoly, shapelypoly)
        skeletonresults.append(SkeletonResult(le_pos_ind, te_ind, rms_error_le_pos))
        if rms_error_le_pos > rms_error_ref:
            search_le_pos -= 1
    while search_le_neg > 0:
        le_neg_ind = (le_neg_ind - 1) % npts
        rms_error_le_neg = objective_function([le_neg_ind, te_ind], sortedPoly, shapelypoly)
        skeletonresults.append(SkeletonResult(le_neg_ind, te_ind, rms_error_le_neg))
        if rms_error_le_neg > rms_error_ref:
            search_le_neg -= 1
    while search_le_neg_te_pos > 0:
        le_neg_te_pos_le_ind = (le_neg_te_pos_le_ind - 1) % npts
        le_neg_te_pos_te_ind = (le_neg_te_pos_te_ind + 1) % npts
        rms_error_le_neg_te_pos = objective_function([le_neg_te_pos_le_ind, le_neg_te_pos_te_ind], sortedPoly, shapelypoly)
        skeletonresults.append(SkeletonResult(le_neg_te_pos_le_ind, le_neg_te_pos_te_ind, rms_error_le_neg_te_pos))
        if rms_error_le_neg_te_pos > rms_error_ref:
            search_le_neg_te_pos -= 1
    while search_le_pos_te_neg > 0:
        le_pos_te_neg_le_ind = (le_pos_te_neg_le_ind + 1) % npts
        le_pos_te_neg_te_ind = (le_pos_te_neg_te_ind - 1) % npts
        rms_error_le_pos_te_neg = objective_function([le_pos_te_neg_le_ind, le_pos_te_neg_te_ind], sortedPoly, shapelypoly)
        skeletonresults.append(SkeletonResult(le_pos_te_neg_le_ind, le_pos_te_neg_te_ind, rms_error_le_pos_te_neg))
        if rms_error_le_pos_te_neg > rms_error_ref:
            search_le_pos_te_neg -= 1
    while search_le_neg_te_neg > 0:
        le_neg_te_neg_le_ind = (le_neg_te_neg_le_ind - 1) % npts
        le_neg_te_neg_te_ind = (le_neg_te_neg_te_ind - 1) % npts
        rms_error_le_neg_te_neg = objective_function([le_neg_te_neg_le_ind, le_neg_te_neg_te_ind], sortedPoly, shapelypoly)
        skeletonresults.append(SkeletonResult(le_neg_te_neg_le_ind, le_neg_te_neg_te_ind, rms_error_le_neg_te_neg))
        if rms_error_le_neg_te_neg > rms_error_ref:
            search_le_neg_te_neg -= 1
    while search_le_pos_te_pos > 0:
        le_pos_te_pos_le_ind = (le_pos_te_pos_le_ind + 1) % npts
        le_pos_te_pos_te_ind = (le_pos_te_pos_te_ind + 1) % npts
        rms_error_le_pos_te_pos = objective_function([le_pos_te_pos_le_ind, le_pos_te_pos_te_ind], sortedPoly, shapelypoly)
        skeletonresults.append(SkeletonResult(le_pos_te_pos_le_ind, le_pos_te_pos_te_ind, rms_error_le_pos_te_pos))
        if rms_error_le_pos_te_pos > rms_error_ref:
            search_le_pos_te_pos -= 1

    best_combination = min(skeletonresults, key=lambda x: x.rms_error)

    le_ind_best, te_ind_best = best_combination.le_ind, best_combination.te_ind
    return le_ind_best, te_ind_best


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
    polygon = shapely.geometry.Polygon(points_2d_closed_refined_voronoi).convex_hull
    is_inside = [shapely.geometry.Point(i).within(polygon) for i in vor.vertices]
    voronoi_sites_inside = vor.vertices[is_inside]

    sort_indices = np.argsort(voronoi_sites_inside[:, 0])
    sites_inside_sorted = voronoi_sites_inside[sort_indices]

    return sites_inside_sorted


def skeletonize_skeleton_sites(points_2d_closed_refined_skeletonize):
    res = len(points_2d_closed_refined_skeletonize)
    dx = np.min(points_2d_closed_refined_skeletonize[:, 0])
    dy = np.min(points_2d_closed_refined_skeletonize[:, 1])

    maxx = np.max(points_2d_closed_refined_skeletonize[:, 0])
    maxy = np.max(points_2d_closed_refined_skeletonize[:, 1])

    scale = max(maxx - dx, maxy - dy)
    factor = res

    px = (points_2d_closed_refined_skeletonize[:, 0] - dx) / scale * factor
    py = (points_2d_closed_refined_skeletonize[:, 1] - dy) / scale * factor

    polygon = np.stack((px, py)).T
    image_size = (res, res)  # Image size
    midline, img = compute_midline(polygon, image_size)

    # Find midline points
    xx_idx, yy_idx = np.where(midline > 0)
    midline_points = np.column_stack((xx_idx, yy_idx))

    mxx = (midline_points[:, 0]) / factor * scale + dy
    myy = (midline_points[:, 1]) / factor * scale + dx

    midline_skeletonize = np.stack([myy, mxx]).T

    return midline_skeletonize


def polygon_to_binary_image(polygon, image_size):
    # Create a blank image
    img = np.zeros(image_size, dtype=np.uint8)

    # Create an array of polygon vertices
    vertices = np.array([polygon], dtype=np.int64)

    # Fill the polygon on the image
    cv2.fillPoly(img, vertices, color=255)

    return img


def compute_midline(polygon, image_size):
    # Convert polygon to binary image
    binary_img = polygon_to_binary_image(polygon, image_size)
    # crop all-zero rows of the binary_img
    binary_img = binary_img[~np.all(binary_img == 0, axis=1)]

    # Apply skeletonization
    skeleton = skeletonize(binary_img, method='lee')

    return skeleton, binary_img


def pointcloud_to_profile(points, res):
    tck, u = splprep(points.T, u=None, s=0.0, per=1, k=3)
    u_new = np.linspace(u.min(), u.max(), res)
    a_new = splev(u_new, tck, der=0)
    points_2d_closed_refined = np.stack([a_new[0], a_new[1]]).T

    return points_2d_closed_refined
