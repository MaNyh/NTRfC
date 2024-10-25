import numpy as np


def test_postprocessing():
    import pyvista as pv
    from ntrfc.turbo.cascade_case.solution import GenericCascadeCase
    from ntrfc.turbo.cascade_case.post import compute_avdr_inout_massave
    from ntrfc.turbo.cascade_case.post import blade_loading_mais
    from ntrfc.turbo.cascade_case.post import blade_loading_cp
    from ntrfc.turbo.cascade_case.utils.domain_utils import Blade2D
    from ntrfc.turbo.airfoil_generators.naca_airfoil_creator import naca

    xs, ys = naca("6510", 256)
    testsolutionpoly = pv.PolyData(np.stack([xs, ys, np.zeros(len(xs))]).T)

    testsolutionpoly.point_data["pMean"] = [1] * testsolutionpoly.number_of_points
    inlet = pv.Plane(direction=(1, 0, 0))
    inlet["u"] = np.ones((inlet.number_of_cells))
    inlet["v"] = np.zeros((inlet.number_of_cells))
    inlet["w"] = np.zeros((inlet.number_of_cells))
    inlet["UMean"] = np.stack([inlet["u"], inlet["v"], inlet["w"]]).T
    inlet["rhoMean"] = np.array([1] * inlet.number_of_cells)
    inlet["pMean"] = np.array([1] * inlet.number_of_cells)
    inlet.ctp()
    outlet = pv.Plane(direction=(-1, 0, 0))
    outlet["u"] = np.ones((outlet.number_of_cells))
    outlet["v"] = np.zeros((outlet.number_of_cells))
    outlet["w"] = np.zeros((outlet.number_of_cells))
    outlet["UMean"] = np.stack([inlet["u"], inlet["v"], inlet["w"]]).T
    outlet["rhoMean"] = np.array([1] * outlet.number_of_cells)
    outlet["pMean"] = np.array([1] * outlet.number_of_cells)
    outlet.ctp()
    blade = Blade2D(testsolutionpoly)
    blade.compute_all_frompoints()
    # Initialize PostProcessing object
    postprocessing = GenericCascadeCase()
    postprocessing.mesh_dict["inlet"] = inlet
    postprocessing.mesh_dict["outlet"] = outlet
    postprocessing.mesh_dict["blade"] = blade.sortedpoints_pv
    postprocessing.blade = blade

    # Test compute_avdr method
    compute_avdr_inout_massave(postprocessing)
    assert postprocessing.avdr == 1

    # Test blade_loading method
    sx, sc, px, pc, _ = blade_loading_cp(postprocessing)
    assert len(sx) == len(sc)
    assert len(px) == len(pc)
    # Test blade_loading method
    blade_loading_mais(postprocessing)
