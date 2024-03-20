import os

from ntrfc.filehandling.datafiles import inplace_change, get_filelist_fromdir, get_directory_structure


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


def test_inplace_change(tmpdir):
    test_oldstring = "old"
    test_newstring = "new"
    test_file = tmpdir / "file.txt"
    with open(test_file, "w") as fobj:
        fobj.write(test_oldstring)

    inplace_change(test_file, test_oldstring, test_newstring)

    with open(test_file, "r") as fobj:
        ans = fobj.read()

    assert ans == test_newstring, "inplace change not working"

    test_oldstring = "old"
    test_fakeoldstring = "nold"
    test_newstring = "new"
    test_file = tmpdir / "file.txt"
    with open(test_file, "w") as fobj:
        fobj.write(test_oldstring)

    inplace_change(test_file, test_fakeoldstring, test_newstring)

    with open(test_file, "r") as fobj:
        ans = fobj.read()

    assert ans == test_oldstring, "inplace change not working"


def test_get_filelist_fromdir(tmpdir):
    os.mkdir(f"{tmpdir}/somedir")

    file_one = "somedir/somefile"
    file_two = "somefile"
    open(f"{tmpdir}/{file_one}", "w")
    open(f"{tmpdir}/{file_two}", "w")
    filelist = get_filelist_fromdir(tmpdir)
    assert len(filelist) == 2
    filebase = [os.path.relpath(i, tmpdir) for i in filelist]
    assert file_one in filebase
    assert file_two in filebase


def test_get_directory_structure(tmpdir):
    # create a directory structure in a temporary directory
    tmpdir.mkdir('root')
    tmpdir.join('root/folder1').mkdir()
    tmpdir.join('root/folder2').mkdir()
    tmpdir.join('root/folder1/subfolder1').mkdir()
    tmpdir.join('root/folder1/subfolder2').mkdir()
    tmpdir.join('root/folder1/subfolder2/file1.txt').write('test')
    tmpdir.join('root/folder1/subfolder2/file2.txt').write('test')
    tmpdir.join('root/folder1/file3.txt').write('test')
    tmpdir.join('root/folder2/file4.txt').write('test')

    # get the directory structure
    directory_structure = get_directory_structure(str(tmpdir.join('root')))

    # check that the directory structure is correct
    assert directory_structure == {
        'root': {'folder1': {'file3.txt': None,
                             'subfolder1': {},
                             'subfolder2': {'file1.txt': None, 'file2.txt': None}},
                 'folder2': {'file4.txt': None}}}
