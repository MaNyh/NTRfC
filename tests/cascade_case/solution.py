import numpy as np
import pyvista as pv


def test_solution(tmpdir):
    from ntrfc.cascade_case.case import solution

    inletname = tmpdir / "fake_inlet.vtk"
    outletname = tmpdir / "fake_outlet.vtk"
    fake_inlet = pv.Plane()
    fake_inlet["U"]=np.array([1]*fake_inlet.number_of_cells)
    fake_inlet["u"] = np.array([1] * fake_inlet.number_of_cells)
    fake_inlet["v"] = np.array([0] * fake_inlet.number_of_cells)
    fake_inlet["w"] = np.array([0] * fake_inlet.number_of_cells)
    fake_inlet["rho"]=np.array([1]*fake_inlet.number_of_cells)
    fake_inlet.save(inletname)
    fake_outlet = pv.Plane()
    fake_outlet["U"]=np.array([1]*fake_outlet.number_of_cells)
    fake_outlet["u"] = np.array([1] * fake_outlet.number_of_cells)
    fake_outlet["v"] = np.array([0] * fake_outlet.number_of_cells)
    fake_outlet["w"] = np.array([0] * fake_outlet.number_of_cells)
    fake_outlet["rho"]=np.array([1]*fake_outlet.number_of_cells)
    fake_outlet.save(outletname)
    case = solution()
    case.read_inlet(inletname)
    case.read_outlet(outletname)
    case.compute_avdr()
    assert case.avdr == 1 , "should be avdr==1"
