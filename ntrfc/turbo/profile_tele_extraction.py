import numpy as np
import pyvista as pv
from scipy.interpolate import splev
from scipy.interpolate import splprep
from shapely.geometry import Polygon, LineString
from skimage.morphology import skeletonize
from ntrfc.geometry.line import lines_from_points
from ntrfc.turbo.pointcloud_methods import extractSidePolys, midline_from_sides
from ntrfc.geometry.alphashape import auto_concaveHull




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

    thicknesses = (lengths_positive+lengths_negative)/2
    reconstructed_pts_positive = centers + thicknesses[:, np.newaxis] * normals
    reconstructed_pts_negative = centers - thicknesses[:, np.newaxis] * normals
    reconstructed_pts = np.concatenate([reconstructed_pts_positive, reconstructed_pts_negative])

    rec_xx,rec_yy ,_= auto_concaveHull(reconstructed_pts[:,0],reconstructed_pts[:,1])

    reconstructed_shapepoly = Polygon(np.stack((rec_xx,rec_yy)).T)
    #compute difference of lengths
    differences = np.abs(lengths_positive - lengths_negative)

    error_thickness = np.mean(differences/thicknesses)
    error_area = reconstructed_shapepoly.symmetric_difference(shapelypoly).area/shapelypoly.area
    error_perimeter = abs(reconstructed_shapepoly.length-shapelypoly.length)/shapelypoly.length
    error_hausdorf = reconstructed_shapepoly.hausdorff_distance(shapelypoly)

    return error_area + error_perimeter + error_hausdorf +error_thickness

def extract_vk_hk(sortedPoly: pv.PolyData) -> (int, int):
    voronoires = 32000

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
    stepsize = len(indices_back_all)*len(indices_front_all)//400
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
    percentile_value = np.percentile(error_data[:, 2], 3)
    lowest10 = error_data[np.where(error_data[:, 2] < percentile_value)]

    front_unique = np.unique(lowest10[:, 0])
    back_unique = np.unique(lowest10[:, 1])

    front_refined = []
    back_refined = []
    for i in front_unique:
        front_refined.append(int(i))
        for j in range(stepsize):
            front_refined.append(int((i + j)%sortedPoly.number_of_points))
            front_refined.append(int((i - j)%sortedPoly.number_of_points))

    for i in back_unique:
        back_refined.append(int(i))
        for j in range(stepsize):
            back_refined.append(int((i + j)%sortedPoly.number_of_points))
            back_refined.append(int((i - j)%sortedPoly.number_of_points))

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
    le_ind_best, te_ind_best = int(best_combination_fine[0]), int(best_combination_fine[1])

    return le_ind_best, te_ind_best





def pointcloud_to_profile(points, res):
    tck, u = splprep(points.T, u=None, s=0.0, per=1, k=3)
    u_new = np.linspace(u.min(), u.max(), res)
    a_new = splev(u_new, tck, der=0)
    points_2d_closed_refined = np.stack([a_new[0], a_new[1]]).T

    return points_2d_closed_refined
