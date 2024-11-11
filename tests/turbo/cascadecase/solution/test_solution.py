def test_generic_cascade_case():
    from ntrfc.turbo.cascade_case.solution import GenericCascadeCase
    from ntrfc.turbo.cascade_case.utils.domain_utils import Blade2D
    import importlib.resources

    inlet_path = importlib.resources.files("ntrfc") / "data/openfoam_cascade_case/boundary/inlet.vtp"
    outlet_path = importlib.resources.files("ntrfc") / "data/openfoam_cascade_case/boundary/outlet.vtp"
    blade_path = importlib.resources.files("ntrfc") / "data/openfoam_cascade_case/boundary/blade_wall.vtp"
    fluid_path = importlib.resources.files("ntrfc") / "data/openfoam_cascade_case/internal.vtu"

    case = GenericCascadeCase()

    case.read_meshes(inlet_path, "inlet")
    case.read_meshes(outlet_path, "outlet")
    case.read_meshes(blade_path, "blade")
    case.read_meshes(fluid_path, "fluid")

    case.set_bladeslice_midz()

    assert isinstance(case.blade, Blade2D)
