def test_calc_dimensionless_gridspacing():
    """
    this method tests the nondimensional gridspacing postprocessing function calc_dimensionless_gridspacing
    a volume mesh will be created. boundary meshes will be extracted of the volume mesh.
    then a simple couette-velocity-field is defined

    calc_dimensionless_gridspacing needs to compute accurately the Delta+-Values

    """
    from ntrfc.postprocessing.turbulence.nondimensionals import calc_dimensionless_gridspacing
    import pyvista as pv
    import numpy as np

    def runtest(height, length, width):
        # 11**3 nodes are enough
        nodes = 11
        # needs to be something simple
        mu_0 = 1  # dynamic viscosity
        rho = 1
        velocity = 2
        gradient = velocity / height
        # analytical solution
        span_x = length / (nodes - 1)
        span_y = width / (nodes - 1)
        span_z = height / (nodes - 1)
        wallshearstress = mu_0 * gradient
        wallshearvelocity = np.sqrt(wallshearstress / rho)
        deltaxplus = wallshearvelocity * span_x / mu_0
        deltayplus = wallshearvelocity * span_y / mu_0
        deltazplus = wallshearvelocity * span_z / mu_0
        # define the mesh
        xrng = np.arange(0, nodes, 1, dtype=np.float32)
        yrng = np.arange(0, nodes, 1, dtype=np.float32)
        zrng = np.arange(0, nodes, 1, dtype=np.float32)
        x, y, z = np.meshgrid(xrng, yrng, zrng)
        grid = pv.StructuredGrid(x, y, z)
        # scale the mesh
        grid.points /= nodes - 1
        grid.points *= np.array([length, height, width])
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
        # define velocityfield
        grid["U"] = np.array([gradient, 0, 0]) * grid.cell_centers().points
        lowerwall["U"] = np.array([gradient, 0, 0]) * lowerwall.cell_centers().points
        upperwall["U"] = np.array([gradient, 0, 0]) * upperwall.cell_centers().points
        grid["rho"] = np.ones(grid.number_of_cells)
        lowerwall["rho"] = np.ones(lowerwall.number_of_cells) * rho
        upperwall["rho"] = np.ones(upperwall.number_of_cells) * rho
        dimless_gridspacings = calc_dimensionless_gridspacing(grid, [lowerwall, upperwall], "U", "rho", mu_0)
        assert all(np.isclose(deltaxplus, dimless_gridspacings[
            "DeltaXPlus"],rtol=0.05)), "calc_dimensionelss_gridspcing in x direction not accurate"
        assert all(np.isclose(deltayplus, dimless_gridspacings[
            "DeltaYPlus"],rtol=0.05)), "calc_dimensionelss_gridspcing in y direction not accurate"
        assert all(np.isclose(deltazplus, dimless_gridspacings[
            "DeltaZPlus"],rtol=0.05)), "calc_dimensionelss_gridspcing in z direction not accurate"


    runtest(height=1, length=1, width=1)
    runtest(height=2, length=1, width=1)
    runtest(height=1, length=2, width=1)
    runtest(height=1, length=1, width=2)

