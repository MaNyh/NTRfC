def test_mesh_expansion():
    from ntrfc.meshquality.meshexpansion import compute_expansion_factors
    import pyvista as pv
    import numpy as np
    grid = pv.ImageData()
    values = np.linspace(0, 10, 1000).reshape((20, 5, 10))
    values.shape
    # Set the grid dimensions: shape + 1 because we want to inject our values on
    #   the CELL data
    grid.dimensions = np.array(values.shape) + 1

    # Edit the spatial reference
    grid.origin = (100, 33, 55.6)  # The bottom left corner of the data set
    grid.spacing = (1, 5, 2)  # These are the cell sizes along each axis

    grid = grid.cast_to_unstructured_grid()

    expansionfactors = compute_expansion_factors(grid)

    assert np.all(expansionfactors == 1)
