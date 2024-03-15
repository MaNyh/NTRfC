import warnings

import numpy as np

from ntrfc.meshquality.standards import classify_mesh_quality


def compute_cell_skewness(grid):
    """
    Compute the skewness of each cell in an unstructured grid.

    Parameters
    ----------
    grid : pyvista.UnstructuredGrid
        The input unstructured grid.

    Returns
    -------
    skewness : list
        A list of the skewness of each cell in the grid.
    """
    qualityname = "Skewness"

    # Compute the skewness of each cell
    skewnesses = []
    for i in range(grid.n_cells):
        cell = grid.extract_cells(i)
        skewnesses.append(skewness(cell.points))
    skewnesses = np.array(skewnesses)

    meshok = classify_mesh_quality(qualityname, skewnesses)

    if not meshok:
        warnings.warn(f"Mesh quality ({qualityname}) is bad, consider refining the mesh.")

    return skewnesses


def skewness(points):
    """
    Compute the skewness of a set of points.

    The skewness is defined as the difference between the maximum and minimum
    distances from the centroid of the points to the vertices of the cell,
    divided by the median distance from the centroid to the vertices of the
    cell.

    Parameters
    ----------
    points : numpy array
        The points of the cell, represented as a 2D array with shape (n, 3),
        where n is the number of points and the columns are the x, y, and z
        coordinates.

    Returns
    -------
    skewness : float
        The skewness of the points.
    """
    # Compute the centroid of the points
    centroid = np.mean(points, axis=0)

    # Compute the distances from the centroid to the vertices of the cell
    distances = np.linalg.norm(points - centroid, axis=1)

    # Compute the skewness as the difference between the maximum and minimum
    # distances, divided by the median distance
    skewness = (np.max(distances) - np.min(distances)) / np.median(distances)

    return skewness
