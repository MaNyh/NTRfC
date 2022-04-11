

def test_massflow_plane():
    import numpy as np
    import pyvista as pv
    from ntrfc.utils.pyvista_utils.plane import massflow_plane

    plane = pv.Plane()
    numcells = plane.number_of_cells
    plane["U"] = plane.cell_normals
    plane["rho"] = np.ones(numcells)

    mflow = np.sum(massflow_plane(plane))

    assert mflow==1.0, "something is wrong"


def test_areaave_plane():
    import numpy as np
    import pyvista as pv
    from ntrfc.utils.pyvista_utils.plane import areaave_plane

    plane = pv.Plane()
    plane["U"] = np.ones(plane.number_of_cells)

    plane_ave = areaave_plane(plane,"U")

    assert plane_ave==1.0, "something is not right"


def test_massflowave_plane():
    import pyvista as pv
    from ntrfc.utils.pyvista_utils.plane import massflowave_plane

    plane = pv.Plane()
    numcells = plane.number_of_cells
    plane["U"] = plane.cell_normals
    plane["rho"] = np.ones(numcells)
    plane["k"] = np.ones(numcells)*2

    assert massflowave_plane(plane,"k")==2.0, "something went wrong"
