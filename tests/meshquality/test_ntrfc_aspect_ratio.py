def test_aspect_ratio():
    from ntrfc.meshquality.aspect_ratio import compute_cell_aspect_ratios
    import pyvista as pv
    import numpy as np

    # test good mesh
    grid = pv.ImageData()
    values = np.linspace(0, 10, 1000).reshape((20, 5, 10))
    values.shape
    # Set the grid dimensions: shape + 1 because we want to inject our values on
    #   the CELL data
    grid.dimensions = np.array(values.shape) + 1

    # Edit the spatial reference
    grid.origin = (100, 33, 55.6)  # The bottom left corner of the data set
    grid.spacing = (1, 5, 2)  # These are the cell sizes along each axis

    aspect_ratios = compute_cell_aspect_ratios(grid)

    assert np.all(aspect_ratios == max(grid.spacing) / min(grid.spacing))

    # test bad mesh
    grid = pv.ImageData()
    values = np.linspace(0, 10, 1000).reshape((20, 5, 10))
    values.shape
    # Set the grid dimensions: shape + 1 because we want to inject our values on
    #   the CELL data
    grid.dimensions = np.array(values.shape) + 1

    # Edit the spatial reference
    grid.origin = (100, 33, 55.6)  # The bottom left corner of the data set
    grid.spacing = (1, 5000, 2)  # These are the cell sizes along each axis

    aspect_ratios = compute_cell_aspect_ratios(grid)

    assert np.all(aspect_ratios == max(grid.spacing) / min(grid.spacing))
