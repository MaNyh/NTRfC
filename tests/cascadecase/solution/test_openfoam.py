def test_solution():
    from ntrfc.cascade_case.solution.openfoam import openfoam_case
    import numpy as np
    import pyvista as pv

    fake_inlet = pv.Plane()
    fake_inlet["U"] = np.array([1] * fake_inlet.number_of_cells)
    fake_inlet["u"] = np.array([1] * fake_inlet.number_of_cells)
    fake_inlet["v"] = np.array([0] * fake_inlet.number_of_cells)
    fake_inlet["w"] = np.array([0] * fake_inlet.number_of_cells)
    fake_inlet["rho"] = np.array([1] * fake_inlet.number_of_cells)

    fake_outlet = pv.Plane()
    fake_outlet["U"] = np.array([1] * fake_outlet.number_of_cells)
    fake_outlet["u"] = np.array([1] * fake_outlet.number_of_cells)
    fake_outlet["v"] = np.array([0] * fake_outlet.number_of_cells)
    fake_outlet["w"] = np.array([0] * fake_outlet.number_of_cells)
    fake_outlet["rho"] = np.array([1] * fake_outlet.number_of_cells)

    case = openfoam_case()
    case.inlet = fake_inlet
    case.outlet = fake_outlet
    case.compute_avdr()
    assert case.avdr == 1, "should be avdr==1"


def test_openfoam_probereading(tmpdir):
    from ntrfc.filehandling.datafiles import create_dirstructure
    from ntrfc.cascade_case.solution.openfoam import openfoam_case

    probefilestring="""# Probe 0 (-0.1599997156611697 -0.1430012427480623 0.02)  # Not Found
# Probe 1 (-0.1411821182118212 -0.1216852152363719 0.02)  # Not Found
# Probe 2 (-0.1223642364236424 -0.1003691877246815 0.02)
# Probe 3 (-0.1035463546354636 -0.07905316021299112 0.02)
# Probe 4 (-0.08472847284728473 -0.05773713270130074 0.02)
# Probe 5 (-0.06591059105910591 -0.03642110518961036 0.02)
# Probe 6 (-0.0470927092709271 -0.01510507767791999 0.02)
# Probe 7 (-0.02827482748274826 0.006210949833770418 0.02)
# Probe 8 (-0.009456945694569452 0.02752697734546078 0.02)
# Probe 9 (0.009192919291929197 0.04894795650267177 0.02)
# Probe 10 (0.02960696069606961 0.06865296732743971 0.02)
# Probe 11 (0.05224722472247226 0.08574924993606114 0.02)
# Probe 12 (0.07652565256525654 0.1004819781869841 0.02)
# Probe 13 (0.1023582358235824 0.1122743312794287 0.02)
# Probe 14 (0.1287368736873687 0.1227380007946636 0.02)
# Probe 15 (0.1551155115511551 0.1331984540812058 0.02)
# Probe 16 (0.1814941494149415 0.143658907367748 0.02)
# Probe 17 (0.2078727872787279 0.1541193606542902 0.02)
# Probe 18 (0.2342514251425143 0.1645798139408323 0.02)
# Probe 19 (0.2599997156611698 0.1747904156361355 0.02)
#                 Probe                       0                       1                       2                       3                       4                       5                       6                       7                       8                       9                      10                      11                      12                      13                      14                      15                      16                      17                      18                      19
#                  Time
    0.03161670583519985                       (-1e+300 -1e+300 -1e+300)                       (-1e+300 -1e+300 -1e+300)                       (46.35401464237102 49.64485123866201 0.009270176741351522)                       (46.79550306530624 50.08611033681288 0.09272118874391876)                       (45.9459533343206 49.87779685922583 0.01462953458600582)                       (46.12027564799758 49.97996423468657 -0.0527782967305729)                       (46.36383372668006 49.89889593844771 -0.155023459154379)                       (45.67672615951206 49.65029568884417 -0.2918631257735307)                       (47.68305274004029 51.3575417928603 -0.1594578663112666)                       (50.35197988441984 47.62926323629534 0.2878495929921642)                       (50.94802741389357 40.49905535044272 """
    dirstructure = ["of_fake_case/postProcessing/Probes/",
                    "of_fake_case/postProcessing/Probes/0.01"]
    create_dirstructure(dirstructure,tmpdir)

    tmpfname =f"{tmpdir}/{dirstructure[1]}/U"
    with open(tmpfname,"w") as fobj:
        fobj.writelines(probefilestring)

    case = openfoam_case()
    case.read_openfoam_probes(f"{tmpdir}/{dirstructure[0]}")
    case.probes.plot_all()
    case.probes.plot_val("U")


def test_read_openfoam_sliceseries(tmpdir):
    from ntrfc.cascade_case.solution.openfoam import openfoam_case
    import pyvista as pv
    import os
    import numpy as np

    noslices = 3
    test_slices = [pv.Plane() for i in range(noslices)]
    slices = []
    ts = []
    post = f"{tmpdir}/postProcessing/"
    slicegroupdir =f"{tmpdir}/postProcessing/some"
    os.mkdir(post)
    os.mkdir(slicegroupdir)
    for idx, slice in enumerate(test_slices):
        slice["U"]=np.zeros(slice.number_of_cells)
        slice = slice.point_data_to_cell_data()
        os.mkdir(f"{slicegroupdir}/{idx}")
        fpath = f"{slicegroupdir}/{idx}/someface.vtk"
        slices.append(fpath)
        slice.save(fpath)
        ts.append(idx)

    test_solution = openfoam_case()
    test_solution.read_openfoam_sliceseries(f"{tmpdir}/postProcessing/some","someface.vtk")
