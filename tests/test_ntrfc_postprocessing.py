import pyvista as pv
import numpy as np

def test_createprofileprobes():
    from ntrfc.postprocessing.turbo.probegeneration import create_profileprobes
    from ntrfc.utils.geometry.airfoil_generators.naca_airfoil_creator import naca
    from ntrfc.utils.geometry.pointcloud_methods import extract_geo_paras
    naca_code = "6009"
    angle = 10  # deg
    res = 240
    xs, ys = naca(naca_code, res, half_cosine_spacing=False)
    sorted_poly = pv.PolyData(np.stack([xs, ys, np.zeros(len(xs))]).T)
    sorted_poly.rotate_z(angle)
    sorted_extracted_poly, psPoly, ssPoly, ind_vk, ind_hk, midsPoly, beta_leading, beta_trailing, camber_angle = extract_geo_paras(sorted_poly,1)
    n_psprobes = 24
    n_ssprobes = 36
    probes_ss, probes_ps = create_profileprobes(ssPoly, psPoly, 1, n_ssprobes, n_psprobes,tolerance=1e-6)
    assert probes_ps.number_of_points == n_psprobes, "number of pressure side probes not correct"
    assert probes_ss.number_of_points == n_ssprobes,  "number of suction side probes not correct"

def test_create_midpassageprobes():
    from ntrfc.postprocessing.turbo.probegeneration import create_midpassageprobes
    from ntrfc.utils.geometry.airfoil_generators.naca_airfoil_creator import naca
    from ntrfc.utils.geometry.pointcloud_methods import extract_geo_paras
    naca_code = "6009"
    angle = 10  # deg
    res = 240
    xs, ys = naca(naca_code, res, half_cosine_spacing=False)
    sorted_poly = pv.PolyData(np.stack([xs, ys, np.zeros(len(xs))]).T)
    sorted_poly.rotate_z(angle)
    sorted_extracted_poly, psPoly, ssPoly, ind_vk, ind_hk, midsPoly, beta_leading, beta_trailing, camber_angle = extract_geo_paras(sorted_poly,1)
    nop = 40
    midspan_probes = create_midpassageprobes(1,-0.3, 0.3,0.1,beta_leading, beta_trailing,midsPoly,nop)
    assert midspan_probes.number_of_points==nop, "number of probes on midpassage line not correct"


def test_getgci():
    # todo FILL
    from ntrfc.postprocessing.meshconvergence.grid_convergece_index import getGCI

    fc3 = 23.263
    fc2 = 23.165
    fc1 = 23.151

    N3 = 18576
    N2 = 74304
    N1 = 297216
    Fs=1.25
    D = 2

    GCI_1, GCI_2, GCI_3 = getGCI(N1, N2, N3, fc1, fc2, fc3, D, Fs=1.25)
