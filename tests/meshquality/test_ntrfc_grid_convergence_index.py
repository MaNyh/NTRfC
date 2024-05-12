def test_getgci():
    from ntrfc.meshquality.grid_convergece_index import getGCI
    import numpy as np
    #
    fc3 = 23.263
    fc2 = 23.165
    fc1 = 23.151

    N3 = 18576
    N2 = 74304
    N1 = 297216

    Fs = 1.25
    D = 2

    GCI_1, GCI_2, GCI_3 = getGCI(N1, N2, N3, fc1, fc2, fc3, D, Fs=Fs)

    assert np.isclose(GCI_1, 0.0001259844787122061), "GCI_1 computation does not seem to be correct"
    assert np.isclose(GCI_2, 0.0008813583711057829), "GCI_3 computation does not seem to be correct"
    assert np.isclose(GCI_3, 0.0061695085977409286), "GCI_3 computation does not seem to be correct"

    # test with bad values

    fc3 = 270.263
    fc2 = 262.165
    fc1 = 10.151

    N3 = 18576
    N2 = 74304
    N1 = 297216

    Fs = 1.25
    D = 2

    getGCI(N1, N2, N3, fc1, fc2, fc3, D, Fs=Fs)
