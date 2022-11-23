import numpy as np
import pyvista as pv
import pytest

import os
ON_CI = 'CI' in os.environ


def test_solution(tmpdir):
    from ntrfc.cascade_case.generic import case

    inletname = tmpdir / "fake_inlet.vtk"
    outletname = tmpdir / "fake_outlet.vtk"
    fake_inlet = pv.Plane()
    fake_inlet["U"] = np.array([1] * fake_inlet.number_of_cells)
    fake_inlet["u"] = np.array([1] * fake_inlet.number_of_cells)
    fake_inlet["v"] = np.array([0] * fake_inlet.number_of_cells)
    fake_inlet["w"] = np.array([0] * fake_inlet.number_of_cells)
    fake_inlet["rho"] = np.array([1] * fake_inlet.number_of_cells)
    fake_inlet.save(inletname)
    fake_outlet = pv.Plane()
    fake_outlet["U"] = np.array([1] * fake_outlet.number_of_cells)
    fake_outlet["u"] = np.array([1] * fake_outlet.number_of_cells)
    fake_outlet["v"] = np.array([0] * fake_outlet.number_of_cells)
    fake_outlet["w"] = np.array([0] * fake_outlet.number_of_cells)
    fake_outlet["rho"] = np.array([1] * fake_outlet.number_of_cells)
    fake_outlet.save(outletname)
    case = case()
    case.read_inlet(inletname)
    case.read_outlet(outletname)
    case.compute_avdr()
    assert case.avdr == 1, "should be avdr==1"

@pytest.mark.skipif(ON_CI, reason="currently cant render on ci-systems")
def test_animations(tmpdir):
    from ntrfc.cascade_case.generic import case
    import pyvista as pv
    import os
    noslices = 3
    test_slices = [pv.Plane() for i in range(noslices)]
    slices = []
    ts = []
    for idx, slice in enumerate(test_slices):
        slice["U"]=np.zeros(slice.number_of_cells)
        slice = slice.point_data_to_cell_data()
        fpath = f"{tmpdir}/{idx}/someface.vtk"
        os.mkdir(f"{tmpdir}/{idx}")
        slices.append(fpath)
        slice.save(fpath)
        ts.append(idx)

    test_solution = case()
    test_solution.sliceseries.add_sliceset(slices,"some",ts)
    test_solution.sliceseries.create_animation("some","U",tmpdir,"U.gif")
