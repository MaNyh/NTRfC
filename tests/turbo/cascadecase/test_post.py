import numpy as np


def test_postprocessing():
    import pyvista as pv
    from ntrfc.turbo.cascade_case.solution import GenericCascadeCase
    from ntrfc.turbo.cascade_case.post import compute_avdr_inout_massave
    from ntrfc.turbo.cascade_case.post import blade_loading_mais
    from ntrfc.turbo.cascade_case.post import blade_loading_cp
    from ntrfc.turbo.cascade_case.post import blade_loading_absolute
    from ntrfc.turbo.cascade_case.utils.domain_utils import Blade2D
    from ntrfc.turbo.airfoil_generators.naca_airfoil_creator import naca

    xs, ys = naca("6510", 256)
    testsolutionpoly = pv.PolyData(np.stack([xs, ys, np.zeros(len(xs))]).T)

    testsolutionpoly.point_data["pMean"] = [1] * testsolutionpoly.number_of_points
    inlet = pv.Plane(direction=(1, 0, 0))
    inlet["u"] = np.ones(inlet.number_of_cells)
    inlet["v"] = np.zeros(inlet.number_of_cells)
    inlet["w"] = np.zeros(inlet.number_of_cells)
    inlet["UMean"] = np.stack([inlet["u"], inlet["v"], inlet["w"]]).T
    inlet["rhoMean"] = np.array([1] * inlet.number_of_cells)
    inlet["pMean"] = np.array([1] * inlet.number_of_cells)
    inlet.ctp()
    outlet = pv.Plane(direction=(-1, 0, 0))
    outlet["u"] = np.ones(outlet.number_of_cells)
    outlet["v"] = np.zeros(outlet.number_of_cells)
    outlet["w"] = np.zeros(outlet.number_of_cells)
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
    postprocessing.case_meta.meanvelocity_name = "UMean"
    postprocessing.case_meta.meandensity_name = "rhoMean"
    postprocessing.case_meta.meanpressure_name = "pMean"

    # Test compute_avdr method
    compute_avdr_inout_massave(postprocessing)
    assert postprocessing.statistics.avdr == 1

    # Test blade_loading method
    _ = blade_loading_cp(postprocessing)
    assert len(postprocessing.statistics.ps_cp) == len(postprocessing.statistics.ps_xc)
    assert len(postprocessing.statistics.ss_cp) == len(postprocessing.statistics.ss_xc)
    # Test blade_loading method

    blade_loading_mais(postprocessing)
    assert len(postprocessing.statistics.ps_mais) == len(postprocessing.statistics.ps_xc)
    assert len(postprocessing.statistics.ss_mais) == len(postprocessing.statistics.ss_xc)

    _ = blade_loading_absolute(postprocessing)
    assert len(postprocessing.statistics.ps_pressure) == len(postprocessing.statistics.ps_xc)
    assert len(postprocessing.statistics.ss_pressure) == len(postprocessing.statistics.ss_xc)

def test_cascade_blade_deltas():
    from ntrfc.turbo.cascade_case.solution import GenericCascadeCase
    from ntrfc.turbo.cascade_case.post import blade_deltas
    import pyvista as pv
    import importlib.resources
    from PIL import Image

    def is_valid_png(file_path):
        try:
            with Image.open(file_path) as img:
                # Check if the format is PNG
                if img.format == 'PNG':
                    return True
                else:
                    return False
        except (IOError, OSError):
            return False

    pv.set_jupyter_backend('trame')

    inlet_path = importlib.resources.files("ntrfc") / "data/openfoam_cascade_case/boundary/inlet.vtp"
    outlet_path = importlib.resources.files("ntrfc") / "data/openfoam_cascade_case/boundary/outlet.vtp"
    blade_path = importlib.resources.files("ntrfc") / "data/openfoam_cascade_case/boundary/blade_wall.vtp"
    fluid_path = importlib.resources.files("ntrfc") / "data/openfoam_cascade_case/internal.vtu"

    case = GenericCascadeCase()

    case.read_meshes(inlet_path, "inlet")
    case.read_meshes(outlet_path, "outlet")
    case.read_meshes(blade_path, "blade")
    case.read_meshes(fluid_path, "fluid")

    case.case_meta.meandensity_name = "rho"
    case.case_meta.meanvelocity_name = "U"
    case.case_meta.meanturbulentkineticenergy_name = "k"
    fig = blade_deltas(case)

    assert is_valid_png(fig)


def test_cascade_case_tux():
    from ntrfc.turbo.cascade_case.solution import GenericCascadeCase
    from ntrfc.turbo.cascade_case.post import cascade_case_tux
    import pyvista as pv
    import importlib.resources
    from PIL import Image

    def is_valid_png(file_path):
        try:
            with Image.open(file_path) as img:
                # Check if the format is PNG
                if img.format == 'PNG':
                    return True
                else:
                    return False
        except (IOError, OSError):
            return False

    pv.set_jupyter_backend('trame')

    inlet_path = importlib.resources.files("ntrfc") / "data/openfoam_cascade_case/boundary/inlet.vtp"
    outlet_path = importlib.resources.files("ntrfc") / "data/openfoam_cascade_case/boundary/outlet.vtp"
    blade_path = importlib.resources.files("ntrfc") / "data/openfoam_cascade_case/boundary/blade_wall.vtp"
    fluid_path = importlib.resources.files("ntrfc") / "data/openfoam_cascade_case/internal.vtu"

    case = GenericCascadeCase()

    case.read_meshes(inlet_path, "inlet")
    case.read_meshes(outlet_path, "outlet")
    case.read_meshes(blade_path, "blade")
    case.read_meshes(fluid_path, "fluid")

    case.case_meta.meandensity_name = "rho"
    case.case_meta.meanvelocity_name = "U"
    case.case_meta.meanturbulentkineticenergy_name = "k"
    fig = cascade_case_tux(case, res=6)

    assert is_valid_png(fig)


def test_cascade_case_meshquality():
    from ntrfc.turbo.cascade_case.solution import GenericCascadeCase
    from ntrfc.turbo.cascade_case.post import cascade_case_meshquality
    import pyvista as pv
    import importlib.resources
    from PIL import Image

    def is_valid_png(file_path):
        try:
            with Image.open(file_path) as img:
                # Check if the format is PNG
                if img.format == 'PNG':
                    return True
                else:
                    return False
        except (IOError, OSError):
            return False

    pv.set_jupyter_backend('trame')

    fluid_path = importlib.resources.files("ntrfc") / "data/openfoam_cascade_case/internal.vtu"

    case = GenericCascadeCase()

    case.read_meshes(fluid_path, "fluid")
    fig = cascade_case_meshquality(case)

    assert is_valid_png(fig[0])
