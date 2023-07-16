import numpy as np
import pyvista as pv
from scipy.spatial import Voronoi



from ntrfc.geometry.line import lines_from_points, polyline_from_points, refine_spline
from ntrfc.geometry.plane import inside_poly
from ntrfc.math.vectorcalc import findNearest, vecDir, vecAbs, compute_minmax_distance_in_pointcloud


def extract_vk_hk(sortedPoly, verbose=False):
    points = sortedPoly.points

    points_2d_closed  = np.vstack((points[:, :2], points[:, :2][0]))
    points_2d_closed_refined = np.stack(refine_spline(points_2d_closed[:, 0], points_2d_closed[:, 1], 100000)).T

    # https://link.springer.com/article/10.1007/s13272-013-0088-6
    # Use the Voronoi algorithm to divide the space into Voronoi cellsy

    vor = Voronoi(points_2d_closed)

    # Identify the center points of the Voronoi cells (also known as Voronoi sites)
    voronoi_sites_inside = vor.vertices[inside_poly(points_2d_closed, vor.vertices)]
    vor_sites_sorted = sorted(zip(voronoi_sites_inside[::, 0],voronoi_sites_inside[::, 1]))

    vor_sites_sorted_interpolated_refined = np.stack(refine_spline(np.array(vor_sites_sorted)[:, 0], np.array(vor_sites_sorted)[:, 1], 6000)).T

    valid_circles = []
    #valid_distances = []

    for vor_site in vor_sites_sorted_interpolated_refined:
        # let the vor_site_profile_distance be the smallest opposingpoint_profile_distance from vor_site to the points
        closest_pt_profile_idx = findNearest(points_2d_closed_refined, vor_site)
        closest_pt_profile = points_2d_closed_refined[closest_pt_profile_idx]

        vor_site_profile_distance = vecAbs(closest_pt_profile - vor_site)
        opposite_pt_profile_tester = vor_site + (vor_site - closest_pt_profile)
        # compute minimal opposingpoint_profile_distance from opposite_pt_profile_tester to the profile
        closest_pt_profile_tester_idx = findNearest(points_2d_closed_refined, opposite_pt_profile_tester)
        opposingpoint_profile_distance = vecAbs(points_2d_closed_refined[closest_pt_profile_tester_idx] - opposite_pt_profile_tester)

        valid_profile_distance = np.isclose((opposingpoint_profile_distance), 0, atol=vor_site_profile_distance *.06)
        if valid_profile_distance:
            valid_circles.append([vor_site[0], vor_site[1]])
            #valid_distances.append(vor_site_profile_distance)

    valid_circles = np.array(valid_circles)

    cama_evenx, cama_eveny = valid_circles[::,0], valid_circles[::,1]

    camberpoints = np.stack([cama_evenx, cama_eveny, np.zeros(len(cama_eveny))]).T
    camberline = lines_from_points(camberpoints)
    LE_camber = camberline.extract_cells(0)
    LE_dir = vecDir(LE_camber.points[-1] - LE_camber.points[0])
    TE_camber = camberline.extract_cells(camberline.number_of_cells - 1)
    TE_dir = vecDir(TE_camber.points[0] - TE_camber.points[-1])
    X = points[::, 0]
    Y = points[::, 1]
    Z = X * 0
    profilepoly = polyline_from_points(np.vstack([np.stack([X, Y, Z]).T, np.stack([X[0], Y[0], Z[0]]).T]))

    chord = compute_minmax_distance_in_pointcloud(sortedPoly.points,minmax="max")

    camber_le_extension = pv.Line(LE_camber.points[0] - LE_dir * chord/10, camberpoints[0], resolution=1000)
    camber_te_extension = pv.Line(camberpoints[-1], TE_camber.points[0] - TE_dir * chord/10, resolution=1000)
    camberline_extended = lines_from_points(np.vstack([camber_le_extension.points,
                                                       camberpoints[1:-2],
                                                       camber_te_extension.points]))

    helpersurface = profilepoly.copy().extrude([0, 0, -1], inplace=True)
    helpersurface = helpersurface.translate([0, 0, .5], inplace=True)

    camberline_computed = camberline_extended.clip_surface(helpersurface, invert=False)

    le_ind = findNearest(np.stack([X, Y, Z]).T, camberline_computed.points[0])
    te_ind = findNearest(np.stack([X, Y, Z]).T, camberline_computed.points[-1])

    # plotter = pv.Plotter()
    # plotter.add_mesh(pv.PolyData(np.vstack([np.stack([X, Y, Z]).T, np.stack([X[0], Y[0], Z[0]]).T])), color='red', label='profile')
    # plotter.add_mesh(camberline, color='blue', line_width=5, label='camberline')
    # for x,y ,r in zip(cama_evenx, cama_eveny, valid_distances):
    #     plotter.add_mesh(pv.Sphere(center=[x,y,0], radius=r).slice(normal="z"), color='black', label='circle',opacity=0.1)
    # #add the voronoi sites
    # plotter.add_mesh(pv.PolyData(np.stack([np.array(vor_sites_sorted_interpolated_refined)[::,0],
    #         np.array(vor_sites_sorted_interpolated_refined)[::,1],
    #         np.zeros(len(vor_sites_sorted_interpolated_refined))]).T), color='green', point_size=10, label='voronoi sites')
    # plotter.add_mesh(camberline)
    # plotter.add_mesh(camber_le_extension)
    # plotter.add_mesh(camber_te_extension)
    # plotter.show()


    # Return the camber line points
    return le_ind, te_ind
