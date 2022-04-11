


def test_massflow_plane():
    import numpy as np
    import pyvista as pv
    from ntrfc.pyvista_utils.surface import massflow_plane

    plane = pv.Plane()
    numcells = plane.number_of_cells
    plane["U"] = plane.cell_normals
    plane["rho"] = np.ones(numcells)

    mflow = massflow_plane(plane)

    assert mflow==1.0, "something is wrong"


def test_areaave_plane():
    import numpy as np
    import pyvista as pv
    from ntrfc.pyvista_utils.surface import areaave_plane

    plane = pv.Plane()
    plane["U"] = np.ones(plane.number_of_cells)

    plane_ave = areaave_plane(plane,"U")

    assert plane_ave==1.0, "something is not right"
