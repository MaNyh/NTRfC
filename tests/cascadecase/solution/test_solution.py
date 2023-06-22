import os

import numpy as np
import pyvista as pv

ON_CI = 'CI' in os.environ


def test_solution(tmpdir):
    from ntrfc.cascade_case.solution import GenericCascadeCase

    inletname = tmpdir / "fake_inlet.vtk"
    outletname = tmpdir / "fake_outlet.vtk"

    fake_inlet = pv.Plane(direction=(1, 0, 0))

    fake_inlet["u"] = np.array([1] * fake_inlet.number_of_cells)
    fake_inlet["v"] = np.array([0] * fake_inlet.number_of_cells)
    fake_inlet["w"] = np.array([0] * fake_inlet.number_of_cells)
    fake_inlet["rho"] = np.array([1] * fake_inlet.number_of_cells)
    fake_inlet["U"] = np.stack([fake_inlet["u"], fake_inlet["v"], fake_inlet["w"]]).T
    fake_inlet.save(inletname)
    fake_outlet = pv.Plane(direction=(-1, 0, 0))

    fake_outlet["u"] = np.array([1] * fake_outlet.number_of_cells)
    fake_outlet["v"] = np.array([0] * fake_outlet.number_of_cells)
    fake_outlet["w"] = np.array([0] * fake_outlet.number_of_cells)
    fake_outlet["rho"] = np.array([1] * fake_outlet.number_of_cells)
    fake_outlet["U"] = np.stack([fake_outlet["u"], fake_outlet["v"], fake_outlet["w"]]).T

    fake_outlet.save(outletname)
    case = GenericCascadeCase()
    case.read_meshes(inletname, "inlet")
    case.read_meshes(outletname, "outlet")
    case.compute_avdr_inout_massave()
    assert case.avdr == 1, "should be avdr==1"


def test_animations(tmpdir):
    from ntrfc.cascade_case.solution import GenericCascadeCase
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


def test_postprocessing():
    import pyvista as pv
    from ntrfc.cascade_case.solution import GenericCascadeCase
    from ntrfc.cascade_case.utils.domain_utils import DomainParameters
    from ntrfc.turbo.airfoil_generators.naca_airfoil_creator import naca

    xs, ys = naca("6510", 256)
    points = pv.PolyData(np.stack([xs, ys, np.zeros(len(xs))]).T)
    domainparas = DomainParameters()
    domainparas.generate_params_by_pointcloud(points)
    domainparas.profile_points["p"] = [1] * domainparas.profile_points.number_of_points

    inlet = pv.Plane()
    inlet["u"] = np.ones((inlet.number_of_cells))
    inlet["v"] = np.zeros((inlet.number_of_cells))
    inlet["w"] = np.zeros((inlet.number_of_cells))
    inlet["U"] = np.stack([inlet["u"], inlet["v"], inlet["w"]]).T
    inlet["rho"] = np.array([1] * inlet.number_of_cells)
    inlet["p"] = np.array([1] * inlet.number_of_cells)
    inlet.ctp()
    outlet = pv.Plane()
    outlet["u"] = np.ones((outlet.number_of_cells))
    outlet["v"] = np.zeros((outlet.number_of_cells))
    outlet["w"] = np.zeros((outlet.number_of_cells))
    outlet["U"] = np.stack([inlet["u"], inlet["v"], inlet["w"]]).T
    outlet["rho"] = np.array([1] * outlet.number_of_cells)
    outlet["p"] = np.array([1] * outlet.number_of_cells)
    outlet.ctp()

    # Initialize PostProcessing object
    postprocessing = GenericCascadeCase()
    postprocessing.mesh_dict["inlet"] = inlet
    postprocessing.mesh_dict["outlet"] = outlet
    postprocessing.domainparams = domainparas

    # Test compute_avdr method
    postprocessing.compute_avdr_inout_massave()
    assert postprocessing.avdr == 1

    # Test blade_loading method
    assert postprocessing.blade_loading() == 0
