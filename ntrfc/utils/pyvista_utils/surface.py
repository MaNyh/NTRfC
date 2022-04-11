import numpy as np
import pyvista as pv
from ntrfc.utils.math.vectorcalc import vecAbs, vecProjection

def calc_dist_from_surface(surface_primary, surface_secondary, verbose=False):
    """
    Distance Between Two Surfaces / A Surface and a Pointcloud
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Compute the average thickness between two surfaces.

    We can compute the thickness between the two surfaces using a few different
    methods. We will demo a method where we compute the normals of the
    bottom surface, and then project a ray to the top surface to compute the
    distance along the surface normals.
    :param surface_primary: pv.UnstructuredGrid
    :param surface_secondary: pv.UnstructuredGrid / pv.PolyData
    :param verbose: plots?
    :return: surface_primary with distances from secondary
    """

    ###############################################################################
    # Ray Tracing Distance
    # ++++++++++++++++++++
    #

    h0n = surface_primary.compute_normals(point_normals=True, cell_normals=False,
                                          auto_orient_normals=True)

    ###############################################################################
    # Travel along normals to the other surface and compute the thickness on each
    # vector.

    h0n["distances"] = np.empty(surface_primary.n_points)
    for i in range(h0n.n_points):
        p = h0n.points[i]
        vec = h0n["Normals"][i] * h0n.length
        p0 = p - vec
        p1 = p + vec
        ip, ic = surface_secondary.ray_trace(p0, p1, first_point=True)
        dist = np.sqrt(np.sum((ip - p) ** 2))
        h0n["distances"][i] = dist

    if any(h0n["distances"] == 0):
        # Replace zeros with nans
        mask = h0n["distances"] == 0
        h0n["distances"][mask] = np.nan
        np.nanmean(h0n["distances"])

    if verbose:
        ###############################################################################
        p = pv.Plotter()
        p.add_mesh(h0n, scalars="distances", smooth_shading=True)
        p.add_mesh(surface_secondary, color=True, opacity=0.75, smooth_shading=True)
        p.show()

    return h0n


def massflow_plane(mesh):
    if not "Normals" in mesh.array_names:
        mesh = mesh.compute_normals()
    if not "Area" in mesh.array_names:
        mesh = mesh.compute_cell_sizes()
    mesh = mesh.point_data_to_cell_data()
    normals = mesh.cell_normals
    rhos = mesh["rho"]
    areas = mesh["Area"]
    velocities = mesh["U"]

    massflow = np.array(
        [vecAbs(vecProjection(velocities[i], normals[i])) for i in range(mesh.number_of_cells)]) ** 2 * rhos * areas

    return np.sum(massflow)


def areaave_plane(mesh, valname):
    array=mesh[valname]
    if not "Area" in mesh.array_names:
        mesh = mesh.compute_cell_sizes()
    areas = mesh["Area"]
    area_ave = np.sum((array.T*areas).T)/np.sum(areas)
    return area_ave


