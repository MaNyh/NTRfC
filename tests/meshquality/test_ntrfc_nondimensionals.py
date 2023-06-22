import numpy as np
import pyvista as pv

from ntrfc.meshquality.nondimensionals import compute_scalar_gradient, constructWallMesh, cellDirections


def test_cellspans():
    from ntrfc.meshquality.nondimensionals import cellSpans
    # 11**3 nodes are enough
    height = 2
    width = 1
    length = 1
    nodes = 11
    # needs to be something simple
    mu_0 = 1  # dynamic viscosity
    rho = 1
    velocity = 2
    gradient = velocity / height
    # analytical solution_utils
    span_x = width / (nodes - 1)
    span_y = height / (nodes - 1)
    span_z = length / (nodes - 1)
    # define the mesh
    grid = structured_testgrid(height, length, nodes, width)

    # define velocityfield
    bounds = grid.bounds
    min_z = bounds[4]
    grid["U"] = [gradient * (grid.cell_centers().points[::, 1][i] - min_z) * np.array([0, 0, 1]) for i in
                 range(grid.number_of_cells)]
    grid["rho"] = np.ones(grid.number_of_cells)
    # extract surface
    surface = grid.extract_surface()
    facecellids = surface.surface_indices()
    upperwallids = []
    lowerwallids = []
    for faceid in facecellids:
        cell = surface.extract_cells(faceid)
        if all(cell.points[::, 1] == 0):
            lowerwallids.append(faceid)
        elif all(cell.points[::, 1] == height):
            upperwallids.append(faceid)
    lowerwall = surface.extract_cells(lowerwallids)
    upperwall = surface.extract_cells(upperwallids)
    lowerwall["U"] = [gradient * (lowerwall.cell_centers().points[::, 1][i] - min_z) * np.array([0, 0, 1]) for i in
                      range(lowerwall.number_of_cells)]
    upperwall["U"] = [gradient * (upperwall.cell_centers().points[::, 1][i] - min_z) * np.array([0, 0, 1]) for i in
                      range(upperwall.number_of_cells)]
    lowerwall["rho"] = np.ones(lowerwall.number_of_cells) * rho
    upperwall["rho"] = np.ones(upperwall.number_of_cells) * rho
    surface = constructWallMesh([lowerwall, upperwall])
    surfacenormals_surface = surface.extract_surface().compute_normals()
    walladjacentids = grid.find_containing_cell(surfacenormals_surface.points)
    volmesh_walladjacent = grid.extract_cells(walladjacentids)

    volmesh_walladjacent["cellCenters"] = volmesh_walladjacent.cell_centers().points
    volmesh_walladjacent["wallNormal"] = [
        surfacenormals_surface.point_data["Normals"][surfacenormals_surface.find_closest_point(i)]
        for i in volmesh_walladjacent.points]
    spans = np.array(cellSpans(volmesh_walladjacent, "U"))

    assert np.all(np.isclose(spans[::, 0], span_x))
    assert np.all(np.isclose(spans[::, 1], span_y))
    assert np.all(np.isclose(spans[::, 2], span_z))


def test_calc_dimensionless_gridspacing():
    """
    this method tests the nondimensional gridspacing postprocessing function calc_dimensionless_gridspacing
    a volume mesh will be created. boundary meshes will be extracted of the volume mesh.
    then a simple couette-velocity-field is defined

    calc_dimensionless_gridspacing needs to compute accurately the Delta+-Values

    """
    from ntrfc.meshquality.nondimensionals import calc_dimensionless_gridspacing
    import numpy as np

    def runtest(height, length, width):
        # 11**3 nodes are enough
        nodes = 11
        # needs to be something simple
        mu_0 = 1  # dynamic viscosity
        rho = 1
        velocity = 2
        gradient = velocity / height
        # analytical solution_utils
        span_x = width / (nodes - 1)
        span_y = height / (nodes - 1)
        span_z = length / (nodes - 1)
        wallshearstress = mu_0 * gradient
        wallshearvelocity = np.sqrt(wallshearstress / rho)
        deltaxplus = wallshearvelocity * span_x / mu_0
        deltayplus = wallshearvelocity * span_y / mu_0
        deltazplus = wallshearvelocity * span_z / mu_0
        grid = structured_testgrid(height, length, nodes, width)

        # define velocityfield
        bounds = grid.bounds
        min_z = bounds[4]
        grid["U"] = [gradient * (grid.cell_centers().points[::, 1][i] - min_z) * np.array([0, 0, 1]) for i in
                     range(grid.number_of_cells)]
        grid["rho"] = np.ones(grid.number_of_cells)
        # extract surface
        surface = grid.extract_surface()
        facecellids = surface.surface_indices()
        upperwallids = []
        lowerwallids = []
        for faceid in facecellids:
            cell = surface.extract_cells(faceid)
            if all(cell.points[::, 1] == 0):
                lowerwallids.append(faceid)
            elif all(cell.points[::, 1] == height):
                upperwallids.append(faceid)
        lowerwall = surface.extract_cells(lowerwallids)
        upperwall = surface.extract_cells(upperwallids)
        lowerwall["U"] = [gradient * (lowerwall.cell_centers().points[::, 1][i] - min_z) * np.array([0, 0, 1]) for i in
                          range(lowerwall.number_of_cells)]
        upperwall["U"] = [gradient * (upperwall.cell_centers().points[::, 1][i] - min_z) * np.array([0, 0, 1]) for i in
                          range(upperwall.number_of_cells)]
        lowerwall["rho"] = np.ones(lowerwall.number_of_cells) * rho
        upperwall["rho"] = np.ones(upperwall.number_of_cells) * rho
        dimless_gridspacings = calc_dimensionless_gridspacing(grid, [lowerwall, upperwall], "U", "rho", mu_0)
        assert all(np.isclose(deltazplus, dimless_gridspacings[
            "DeltaXPlus"], rtol=0.05)), "calc_dimensionelss_gridspcing in x direction not accurate"
        assert all(np.isclose(deltayplus, dimless_gridspacings[
            "DeltaYPlus"], rtol=0.05)), "calc_dimensionelss_gridspcing in y direction not accurate"
        assert all(np.isclose(deltaxplus, dimless_gridspacings[
            "DeltaZPlus"], rtol=0.05)), "calc_dimensionelss_gridspcing in z direction not accurate"

    runtest(height=1, length=1, width=1)
    runtest(height=2, length=1, width=1)
    runtest(height=1, length=2, width=1)
    runtest(height=1, length=1, width=2)


def structured_testgrid(height, length, nodes, width):
    # define the mesh
    xrng = np.arange(0, nodes, 1, dtype=np.float32)
    yrng = np.arange(0, nodes, 1, dtype=np.float32)
    zrng = np.arange(0, nodes, 1, dtype=np.float32)
    x, y, z = np.meshgrid(xrng, yrng, zrng)
    grid = pv.StructuredGrid(x, y, z)
    # scale the mesh
    grid.points /= nodes - 1
    grid.points *= np.array([width, height, length])
    return grid


def test_calc_dimensionless_gridspacing():
    """
    this method tests the nondimensional gridspacing postprocessing function calc_dimensionless_gridspacing
    a volume mesh will be created. boundary meshes will be extracted of the volume mesh.
    then a simple couette-velocity-field is defined

    calc_dimensionless_gridspacing needs to compute accurately the Delta+-Values

    """
    from ntrfc.meshquality.nondimensionals import calc_dimensionless_yplus
    import numpy as np

    def runtest(height, length, width):
        # 11**3 nodes are enough
        nodes = 11
        # needs to be something simple
        mu_0 = 1  # dynamic viscosity
        rho = 1
        velocity = 2
        gradient = velocity / height
        # analytical solution_utils
        span_x = width / (nodes - 1)
        span_y = height / (nodes - 1)
        span_z = length / (nodes - 1)
        wallshearstress = mu_0 * gradient
        wallshearvelocity = np.sqrt(wallshearstress / rho)
        deltayplus = wallshearvelocity * span_y / 2 / mu_0
        # define the mesh
        grid = structured_testgrid(height, length, nodes, width)

        # define velocityfield
        bounds = grid.bounds
        min_z = bounds[4]
        grid["U"] = [gradient * (grid.cell_centers().points[::, 1][i] - min_z) * np.array([0, 0, 1]) for i in
                     range(grid.number_of_cells)]
        grid["rho"] = np.ones(grid.number_of_cells)
        # extract surface
        surface = grid.extract_surface()
        facecellids = surface.surface_indices()
        upperwallids = []
        lowerwallids = []
        for faceid in facecellids:
            cell = surface.extract_cells(faceid)
            if all(cell.points[::, 1] == 0):
                lowerwallids.append(faceid)
            elif all(cell.points[::, 1] == height):
                upperwallids.append(faceid)
        lowerwall = surface.extract_cells(lowerwallids)
        upperwall = surface.extract_cells(upperwallids)
        lowerwall["U"] = [gradient * (lowerwall.cell_centers().points[::, 1][i] - min_z) * np.array([0, 0, 1]) for i in
                          range(lowerwall.number_of_cells)]
        upperwall["U"] = [gradient * (upperwall.cell_centers().points[::, 1][i] - min_z) * np.array([0, 0, 1]) for i in
                          range(upperwall.number_of_cells)]
        lowerwall["rho"] = np.ones(lowerwall.number_of_cells) * rho
        upperwall["rho"] = np.ones(upperwall.number_of_cells) * rho
        dimless_gridspacings = calc_dimensionless_yplus(grid, [lowerwall, upperwall], "U", "rho", mu_0)
        assert all(np.isclose(deltayplus, dimless_gridspacings[
            "yPlus"], rtol=0.05)), "calc_dimensionelss_gridspcing in x direction not accurate"

    runtest(height=1, length=1, width=1)
    runtest(height=2, length=1, width=1)
    runtest(height=1, length=2, width=1)
    runtest(height=1, length=1, width=2)


def test_compute_wallshearstress():
    from ntrfc.meshquality.nondimensionals import get_wall_shear_stress_velocity
    # 11**3 nodes are enough
    nodes = 11
    # needs to be something simple
    mu_0 = 1  # dynamic viscosity
    rho = 1
    velocity = 2
    height = 2
    length = 1
    width = 1
    gradient = velocity / height
    # analytical solution_utils
    wallshearstress = mu_0 * gradient
    wallshearvelocity = np.sqrt(wallshearstress / rho)

    # define the mesh
    grid = structured_testgrid(height, length, nodes, width)

    # define velocityfield
    bounds = grid.bounds
    min_z = bounds[4]
    grid["U"] = [gradient * (grid.cell_centers().points[::, 1][i] - min_z) * np.array([0, 0, 1]) for i in
                 range(grid.number_of_cells)]
    grid["rho"] = np.ones(grid.number_of_cells)
    grid = compute_scalar_gradient(grid, "U")

    # extract surface
    surface = grid.extract_surface()
    facecellids = surface.surface_indices()
    upperwallids = []
    lowerwallids = []
    for faceid in facecellids:
        cell = surface.extract_cells(faceid)
        if all(cell.points[::, 1] == 0):
            lowerwallids.append(faceid)
        elif all(cell.points[::, 1] == height):
            upperwallids.append(faceid)
    lowerwall = surface.extract_cells(lowerwallids)
    upperwall = surface.extract_cells(upperwallids)

    lowerwall["U"] = [gradient * (lowerwall.cell_centers().points[::, 1][i] - min_z) * np.array([0, 0, 1]) for i in
                      range(lowerwall.number_of_cells)]
    upperwall["U"] = [gradient * (upperwall.cell_centers().points[::, 1][i] - min_z) * np.array([0, 0, 1]) for i in
                      range(upperwall.number_of_cells)]
    lowerwall["rho"] = np.ones(lowerwall.number_of_cells) * rho
    upperwall["rho"] = np.ones(upperwall.number_of_cells) * rho

    surface = constructWallMesh([lowerwall, upperwall])
    surfacenormals_surface = surface.extract_surface().compute_normals()

    walladjacentids = grid.find_containing_cell(surfacenormals_surface.points)
    volmesh_walladjacent = grid.extract_cells(walladjacentids)

    volmesh_walladjacent["cellCenters"] = volmesh_walladjacent.cell_centers().points
    volmesh_walladjacent["wallNormal"] = [
        surfacenormals_surface.point_data["Normals"][surfacenormals_surface.find_closest_point(i)]
        for i in volmesh_walladjacent.points]

    utaus = get_wall_shear_stress_velocity(volmesh_walladjacent, mu_0, "rho", "U")

    assert np.all(np.isclose(utaus, wallshearvelocity))


def test_compute_scalar_gradient():
    # 11**3 nodes are enough
    nodes = 11
    # needs to be something simple
    height = 1
    length = 1
    width = 1
    velocity = 2
    gradient = velocity / height

    grid = structured_testgrid(height, length, nodes, width)
    # define velocityfield
    bounds = grid.bounds
    min_z = bounds[4]
    grid["U"] = [gradient * (grid.cell_centers().points[::, 1][i] - min_z) * np.array([0, 0, 1]) for i in
                 range(grid.number_of_cells)]
    expected = np.array([[0., 0., 0.],
                         [0., 0., 0.],
                         [0., 2., 0.]])
    # Compute the gradient and test it
    mesh = compute_scalar_gradient(grid, "U")
    grad_array = mesh[f"grad_U"].reshape(mesh.number_of_cells, 3, 3)
    assert np.all([np.all(np.isclose(expected, i)) for i in grad_array])


def test_constructWallMesh():
    # Create some test surfaces to merge
    surface_1 = pv.Plane()
    surface_2 = pv.Plane((1, 1, 1))

    # Call the function with the test surfaces
    result = constructWallMesh([surface_1, surface_2])

    # Check that the result is a valid UnstructuredGrid object
    assert isinstance(result, pv.UnstructuredGrid)

    # Check that the result has the expected number of points and cells
    assert result.n_points == 242
    assert result.n_cells == 200


def test_cellDirections():
    # Test 1: Check that unit vectors are returned correctly
    cellUMean = np.array([1, 2, 3])
    wallNorm = np.array([4, 5, 6])
    expected = np.array([[0.26726124, 0.53452248, 0.80178373], [0.45584231, 0.56980288, 0.68376346],
                         [-0.40824829, 0.81649658, -0.40824829]])
    actual = cellDirections(cellUMean, wallNorm)
    assert np.all(np.isclose(actual, expected))
