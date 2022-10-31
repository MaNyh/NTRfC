import os


def test_yamlDictRead(tmpdir):
    """
    tests if yaml is returning a known dictionary
    """
    from ntrfc.filehandling.datafiles import yaml_dict_read

    test_file = tmpdir / "test.yaml"
    with open(test_file, "w") as handle:
        handle.write("test_key: True\n")
    assert yaml_dict_read(test_file) == {"test_key": True}


def test_yamlDictWrite(tmpdir):
    """
    tests if yaml is writing and returning a known dictionary
    """
    from ntrfc.filehandling.datafiles import yaml_dict_read, write_yaml_dict

    test_file = tmpdir / "test.yaml"
    test_dict = {"test_key": True}
    write_yaml_dict(test_file, test_dict)

    assert yaml_dict_read(test_file) == test_dict


def test_pickle_operations(tmpdir):
    """
    checks if the pickle-operators are working
    :param tmpdir:
    :return:
    """
    from ntrfc.filehandling.datafiles import write_pickle, read_pickle, write_pickle_protocolzero

    fname = tmpdir / "test.pkl"
    dict = {"test": 1}
    write_pickle(fname, dict)
    pklread = read_pickle(fname)
    assert dict["test"] == pklread["test"]
    write_pickle_protocolzero(fname, dict)
    pklread_zero = read_pickle(fname)
    assert dict["test"] == pklread_zero["test"]


def test_create_dirstructure(tmpdir):
    from ntrfc.filehandling.datafiles import create_dirstructure

    dirstructure = ["ast/bla", "ast/ble"]
    create_dirstructure(dirstructure, tmpdir)
    checks = [os.path.isdir(path) for path in [os.path.join(tmpdir, relpath) for relpath in dirstructure]]
    assert all(checks), "not all directories have been created"
