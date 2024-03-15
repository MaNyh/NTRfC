import os

import numpy as np
import pyvista as pv

ON_CI = 'CI' in os.environ


def test_solution(tmpdir):
    from ntrfc.turbo.cascade_case.solution import GenericCascadeCase
    from ntrfc.turbo.cascade_case.post import compute_avdr_inout_massave

    inletname = tmpdir / "fake_inlet.vtk"
    outletname = tmpdir / "fake_outlet.vtk"

    fake_inlet = pv.Plane(direction=(1, 0, 0))

    fake_inlet["u"] = np.array([1] * fake_inlet.number_of_cells)
    fake_inlet["v"] = np.array([0] * fake_inlet.number_of_cells)
    fake_inlet["w"] = np.array([0] * fake_inlet.number_of_cells)
    fake_inlet["rhoMean"] = np.array([1] * fake_inlet.number_of_cells)
    fake_inlet["UMean"] = np.stack([fake_inlet["u"], fake_inlet["v"], fake_inlet["w"]]).T
    fake_inlet.save(inletname)
    fake_outlet = pv.Plane(direction=(-1, 0, 0))

    fake_outlet["u"] = np.array([1] * fake_outlet.number_of_cells)
    fake_outlet["v"] = np.array([0] * fake_outlet.number_of_cells)
    fake_outlet["w"] = np.array([0] * fake_outlet.number_of_cells)
    fake_outlet["rhoMean"] = np.array([1] * fake_outlet.number_of_cells)
    fake_outlet["UMean"] = np.stack([fake_outlet["u"], fake_outlet["v"], fake_outlet["w"]]).T

    fake_outlet.save(outletname)
    case = GenericCascadeCase()
    case.read_meshes(inletname, "inlet")
    case.read_meshes(outletname, "outlet")
    compute_avdr_inout_massave(case)
    assert case.avdr == 1, "should be avdr==1"


def test_animations(tmpdir):
    from ntrfc.turbo.cascade_case.solution import GenericCascadeCase
    import pyvista as pv

    if ON_CI:
        pv.start_xvfb()

    noslices = 3
    test_slices = [pv.Plane() for i in range(noslices)]
    slices = []
    ts = []
    for idx, slice in enumerate(test_slices):
        slice["U"] = np.zeros(slice.number_of_cells)
        slice = slice.point_data_to_cell_data()
        fpath = f"{tmpdir}/{idx}/someface.vtk"
        os.mkdir(f"{tmpdir}/{idx}")
        slices.append(fpath)
        slice.save(fpath)
        ts.append(idx)

    test_solution = GenericCascadeCase()
    test_solution.sliceseries.add_sliceset(slices, "some", ts)
    test_solution.sliceseries.create_animation("some", "U", tmpdir, "U.gif")
