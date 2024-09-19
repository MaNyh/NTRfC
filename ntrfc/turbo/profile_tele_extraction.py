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
from ntrfc.geometry.alphashape import auto_concaveHull

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



def periodic_weights(length, ind_1, ind_2):
    """
    Generate a periodic signal of given length with maxima at ind_1 and ind_2
    and minima in between, with all values between 0 and 1.

    Parameters:
    - length: Total length of the signal.
    - ind_1: Index of the first maximum (must be less than ind_2).
    - ind_2: Index of the second maximum (must be greater than ind_1).

    Returns:
    - A numpy array containing the periodic signal.
    """

    # Create an array of zeros
    signal = np.zeros(length)

    # 1. Rise from 0 to maximum at ind_1
    slope_rise_1 = 1 / ind_1
    for i in range(ind_1):
        signal[i] = slope_rise_1 * i  # Linear rise from 0 to 1

    # 2. Fall from maximum at ind_1 to minimum (0) at midpoint between ind_1 and ind_2
    mid_point = (ind_1 + ind_2) // 2
    slope_fall = -1 / (mid_point - ind_1)
    for i in range(ind_1, mid_point):
        signal[i] = 1 + slope_fall * (i - ind_1)  # Linear fall from 1 to 0

    # 3. Rise again from minimum (0) to maximum at ind_2
    slope_rise_2 = 1 / (ind_2 - mid_point)
    for i in range(mid_point, ind_2):
        signal[i] = slope_rise_2 * (i - mid_point)  # Linear rise from 0 to 1

    # 4. Fall to 0 by the end (to ensure periodicity)
    slope_fall_2 = -1 / (length - ind_2)
    for i in range(ind_2, length):
        signal[i] = 1 + slope_fall_2 * (i - ind_2)  # Linear fall from 1 to 0

    return signal


import numpy as np
from scipy.interpolate import CubicSpline


def curvature_spline(points):
    # Punkte auf x- und y-Koordinaten aufteilen
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]

    # Spline-Interpolation mit scipy
    cs_x = CubicSpline(np.arange(len(x)), x)
    cs_y = CubicSpline(np.arange(len(y)), y)

    # Erste und zweite Ableitung des Splines berechnen
    t = np.linspace(0, len(x) - 1, len(points))  # Parameterwerte für den Spline (z.B. 1000 Schritte)
    dx = cs_x(t, 1)
    dy = cs_y(t, 1)
    ddx = cs_x(t, 2)
    ddy = cs_y(t, 2)

    # Krümmung für jeden Punkt berechnen
    curvature = np.abs(dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** (3 / 2)

    return curvature



def compute_midline_error(ind_1_start, ind_2_start, sortedPoly, shapelypoly):
    ps_pv, ss_pv = extractSidePolys(ind_1_start, ind_2_start, sortedPoly)

    mid = lines_from_points(midline_from_sides(ps_pv, ss_pv, res=50).points)

    mid_normals = mid.extrude([0, 0, 1]).compute_normals().slice("z").compute_cell_sizes()


    normals = mid_normals["Normals"]
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

    thicknesses = np.min(np.stack([lengths_positive, lengths_negative]).T, axis=1)
    reconstructed_pts_positive = centers + thicknesses[:, np.newaxis] * normals
    reconstructed_pts_negative = centers - thicknesses[:, np.newaxis] * normals
    reconstructed_pts = np.concatenate([reconstructed_pts_positive, reconstructed_pts_negative])

    rec_xx,rec_yy ,_= auto_concaveHull(reconstructed_pts[:,0],reconstructed_pts[:,1])

    reconstructed_shapepoly = Polygon(np.stack((rec_xx,rec_yy)).T)

    #error_thickness = np.mean(differences/thicknesses)
    error_area = reconstructed_shapepoly.symmetric_difference(shapelypoly).area/shapelypoly.area
    error_perimeter = abs(reconstructed_shapepoly.length-shapelypoly.length)/shapelypoly.length
    error_hausdorf = reconstructed_shapepoly.hausdorff_distance(shapelypoly)

    return error_area * error_perimeter * error_hausdorf

def extract_vk_hk(sortedPoly: pv.PolyData) -> (int, int):
    voronoires = 64000

    points_orig = sortedPoly.points
    points_2d_closed_refined_voronoi = pointcloud_to_profile(points_orig, voronoires)

    min_x_idx = np.argmin(sortedPoly.points[:, 0])
    max_x_idx = np.argmax(sortedPoly.points[:, 0])
    min_x = sortedPoly.points[min_x_idx][0]
    max_x = sortedPoly.points[max_x_idx][0]

    delta_x = max_x - min_x
    test_range = 0.050

    indices_front_all = np.where(sortedPoly.points[:, 0] < min_x + delta_x * (test_range))[0]
    indices_back_all = np.where(sortedPoly.points[:, 0] > max_x - delta_x * (test_range))[0]
    stepsize = len(indices_back_all)*len(indices_front_all)//300
    indices_front_coarse = indices_front_all[::stepsize]
    indices_back_coarse = indices_back_all[::stepsize]

    shapelypoly = Polygon(points_2d_closed_refined_voronoi)

    specific_combinations = np.array(np.meshgrid(indices_front_coarse, indices_back_coarse)).T.reshape(-1, 2)

    error_data = []

    # Evaluate the objective function for each combination
    for indices in specific_combinations:
        le_ind, te_ind = int(indices[0]), int(indices[1])
        rms_error = compute_midline_error(le_ind, te_ind, sortedPoly, shapelypoly)
        error_data.append((*indices, rms_error))

    error_data=np.array(error_data)

    # get best 10% of combinations
    arr = np.argsort(error_data[:, 2])[:int(len(error_data) * 0.1)]
    arr.sort()
    arr_indices = error_data[arr][:, :2].astype(int)
    front_unique = np.unique(arr_indices[:, 0])
    back_unique = np.unique(arr_indices[:, 1])

    front_refined = []
    back_refined = []
    for i in front_unique:
        front_refined.append(i)
        for j in range(stepsize):
            front_refined.append((i + j)%sortedPoly.number_of_points)
            front_refined.append((i - j)%sortedPoly.number_of_points)

    for i in back_unique:
        back_refined.append(i)
        for j in range(stepsize):
            back_refined.append((i + j)%sortedPoly.number_of_points)
            back_refined.append((i - j)%sortedPoly.number_of_points)

    front_refined = np.unique(np.array(front_refined))
    back_refined = np.unique(np.array(back_refined))

    specific_combinations_fine = np.array(np.meshgrid(front_refined, back_refined)).T.reshape(-1, 2)

    error_data_fine = []

    # Evaluate the objective function for each combination

    for indices in specific_combinations_fine:
        le_ind, te_ind = int(indices[0]), int(indices[1])
        rms_error = compute_midline_error(le_ind, te_ind, sortedPoly, shapelypoly)
        error_data_fine.append((*indices, rms_error))

    error_data_fine = np.array(error_data_fine)
    best_combination_fine = specific_combinations_fine[np.argmin(error_data_fine, axis=0)[2]]


    le_ind_best, te_ind_best = best_combination_fine[0], best_combination_fine[1]

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
