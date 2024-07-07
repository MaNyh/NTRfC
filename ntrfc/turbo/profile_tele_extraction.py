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
from tqdm import tqdm

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
        # Compute the intersection with the polygon of all rays
        intersections_positive = [LineString([centers[i], ray_positive[i]]).intersection(shapelypoly) for i in range(len(centers))]
        intersections_negative = [LineString([centers[i], ray_negative[i]]).intersection(shapelypoly) for i in range(len(centers))]


    inlier_indices = np.where(inliers_turkey * inliers_zscore * inliers_mad)[0]

    valid_midline = skeletonize_sites[inlier_indices]
    return valid_midline


def extract_vk_hk(sortedPoly: pv.PolyData) -> (int, int):
    voronoires = 64000

    points_orig = sortedPoly.points
    npts = len(points_orig)
    points_2d_closed_refined_voronoi = pointcloud_to_profile(points_orig, voronoires)

    #create sheply polygon from refined pointset
    shapelypoly = Polygon(points_2d_closed_refined_voronoi)
    raylength = 1000
    # todo; estimate le and te regions, only check combinations of these

    min_x = np.min(points_orig[:, 0])
    max_x = np.max(points_orig[:, 0])

    le_est_xmax = min_x + 0.01 * (max_x - min_x)
    te_est_xmin = max_x - 0.01 * (max_x - min_x)

    le_indices = np.where(points_orig[:, 0] < le_est_xmax)[0]
    te_indices = np.where(points_orig[:, 0] > te_est_xmin)[0]


    combinations = [(i, j) for i in le_indices for j in te_indices]



    rms_errors = []
    #max_errors = []

    midsp = []
    for i, j in tqdm(combinations):
        ind_1_start = i
        ind_2_start = j

        ps_pv, ss_pv = extractSidePolys(ind_1_start,ind_2_start, sortedPoly)

        mid = lines_from_points(midline_from_sides(ps_pv, ss_pv,res=200).points)

        mid_normals = mid.extrude([0,0,1]).compute_normals().slice("z").compute_cell_sizes()

        normals = mid_normals["Normals"]
        lengths = mid_normals["Length"]
        centers = mid_normals.cell_centers().points

        ray_positive = centers + raylength * normals
        ray_negative = centers - raylength * normals

        # Compute the intersection with the polygon of all rays
        intersections_positive = [LineString([centers[i], ray_positive[i]]).intersection(shapelypoly) for i in range(len(centers))]
        intersections_negative = [LineString([centers[i], ray_negative[i]]).intersection(shapelypoly) for i in range(len(centers))]

        # compute length of each intersection
        lengths_positive = np.array([line.length for line in intersections_positive])
        lengths_negative = np.array([line.length for line in intersections_negative])

        differences = np.abs(lengths_positive - lengths_negative) /((lengths_positive + lengths_negative)/2)

        rms_error = np.sqrt(np.mean(differences**2))

        rms_errors.append(rms_error)
       # max_errors.append(np.max(differences))

        # mid_normals["diff"]=differences
        # midsp.append(mid_normals)

        # verbose = False
        # if verbose:
        #     p = pv.Plotter()
        #     mid_normals = mid_normals.translate([0, 0, -mid_normals.bounds[5]])
        #     p.add_mesh(mid_normals, scalars="diff")
        #     p.add_mesh(ps_pv, color="red")
        #     p.add_mesh(ss_pv, color="green")
        #     for ineg in intersections_negative:
        #         p.add_mesh(pv.Line(*np.array(ineg.coords) - ineg.coords[0][2]), color="blue")
        #     for ipos in intersections_positive:
        #         p.add_mesh(pv.Line(*np.array(ipos.coords) - ipos.coords[0][2]), color="green")
        #     p.add_mesh(pv.PolyData(np.zeros(3)), color="black")
        #     p.view_xy()
        #     p.show_axes()
        #     p.show()


    inds = combinations[np.argmin(rms_errors)]
    le_ind = inds[np.argmin(points_orig[inds][:, 0])]
    te_ind = inds[np.argmax(points_orig[inds][:, 0])]

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

    # Apply skeletonization
    skeleton = skeletonize(binary_img, method='lee')

    return skeleton, binary_img


def pointcloud_to_profile(points, res):
    tck, u = splprep(points.T, u=None, s=0.0, per=1, k=3)
    u_new = np.linspace(u.min(), u.max(), res)
    a_new = splev(u_new, tck, der=0)
    points_2d_closed_refined = np.stack([a_new[0], a_new[1]]).T

    return points_2d_closed_refined
