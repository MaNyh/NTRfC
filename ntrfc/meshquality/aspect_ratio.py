import warnings

import numpy as np
import pyvista as pv
import vtk

from ntrfc.meshquality.standards import classify_mesh_quality


def compute_cell_aspect_ratios(grid: pv.UnstructuredGrid) -> np.ndarray:
    """Compute the aspect ratio of each cell in an unstructured grid.

    The aspect ratio of a cell is defined as the ratio of the longest edge length
    to the shortest edge length of the cell.

    Parameters:
        grid (pv.UnstructuredGrid): The unstructured grid.

    Returns:
        np.ndarray: An array of aspect ratios, one for each cell in the grid.
    """
    qualityname = "AspectRatio"

    cellids = range(0, grid.number_of_cells)
    # Compute the edge lengths for each cell
    aspect_ratios = []
    vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)
    for cellid in cellids:
        # Get the indices of the points that make up the cell
        cell = grid.extract_cells(cellid)

        edges = cell.extract_all_edges().compute_cell_sizes(length=True, area=False, volume=False)

        aspect_ratios.append(max(edges["Length"]) / min(edges["Length"]))

    aspect_ratios = np.array(aspect_ratios)

    meshok = classify_mesh_quality(qualityname, aspect_ratios)

    if not meshok:
        warnings.warn(f"Mesh quality ({qualityname}) is bad, consider refining the mesh.")

    return aspect_ratios
