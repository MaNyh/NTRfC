import numpy as np
from matplotlib import path as mpltPath

from ntrfc.math.vectorcalc import vecAbs, vecProjection


def massflow_plane(mesh, rhoname="rho", velocityname="U"):
    normals = mesh.compute_normals(point_normals=False, cell_normals=True)["Normals"]
    rhos = mesh.cell_data[rhoname]
    areas = mesh.compute_cell_sizes()["Area"]
    velocities = mesh.cell_data[velocityname]

    massflow = np.array(
        [vecAbs(vecProjection(velocities[i], normals[i])) for i in range(mesh.number_of_cells)]) ** 2 * rhos * areas

    return massflow


def massflowave_plane(mesh, valname, rhoname="rho", velocityname="U"):
    massflow = massflow_plane(mesh, rhoname=rhoname, velocityname=velocityname)

    mass_ave = np.nansum(mesh[valname] * massflow) / np.nansum(massflow)

    return mass_ave


def areaave_plane(mesh, valname):
    array = mesh[valname]
    if "Area" not in mesh.array_names:
        mesh = mesh.compute_cell_sizes()
    areas = mesh["Area"]
    area_ave = np.sum((array.T * areas).T) / np.sum(areas)
    return area_ave


def inside_poly(polygon, points):
    """
    :param polygon: (x,y)
    :param points: (x,y)
    :return: list of True or False for indicating weather inside or not
    """
    path = mpltPath.Path(polygon)
    inside = path.contains_points(points)
    return inside
