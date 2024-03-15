import warnings

import numpy as np
import pyvista as pv

from ntrfc.meshquality.standards import classify_mesh_quality


def compute_expansion_factors(grid: pv.UnstructuredGrid) -> np.ndarray:
    qualityname = "MeshExpansion"

    # Compute cell volumes
    cell_volumes = grid.compute_cell_sizes()["Volume"]

    # Number of cells in the grid
    n_cells = grid.n_cells

    # Initialize array to store expansion factors
    expansion_factors = np.zeros(n_cells)

    # Iterate over cells in the grid
    for i in range(n_cells):
        # Find the indices of the neighbors of the current cell
        neighbor_indices = get_neighbor_cell_ids(grid, i)

        # Find the volume of the largest neighboring cell
        max_neighbor_volume = max([cell_volumes[j] for j in neighbor_indices])

        # Compute expansion factor for the current cell
        expansion_factors[i] = max_neighbor_volume / cell_volumes[i]

    expansion_factors = np.array(expansion_factors)

    meshok = classify_mesh_quality(qualityname, expansion_factors)

    if not meshok:
        warnings.warn(f"Mesh quality ({qualityname}) is bad, consider refining the mesh.")

    return expansion_factors


def get_neighbor_cell_ids(grid, cell_idx):
    """Helper to get neighbor cell IDs."""
    cell = grid.GetCell(cell_idx)
    pids = pv.vtk_id_list_to_array(cell.GetPointIds())
    neighbors = set(grid.extract_points(pids)["vtkOriginalCellIds"])
    neighbors.discard(cell_idx)
    return np.array(list(neighbors))
