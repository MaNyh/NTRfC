import os
import pickle
from functools import reduce
from pathlib import Path

import yaml


def yaml_dict_read(yml_file):
    args_from_yaml = {}

    with open(yml_file, "r", newline='') as Fobj:
        document = yaml.load_all(Fobj, Loader=yaml.FullLoader)
        for settings in document:
            for key, value in settings.items():
                args_from_yaml[key] = value
    return args_from_yaml


def write_pickle_protocolzero(file, args):
    with open(file, "wb") as Fobj:
        pickle.dump(args, Fobj, protocol=0)


def write_pickle(file, args):
    with open(file, "wb") as Fobj:
        pickle.dump(args, Fobj)


def read_pickle(file):
    with open(file, 'rb') as f:
        args = pickle.load(f)
    return args


def write_yaml_dict(fpath, data):
    """
    :param fpath: target path
    :param data: dictionary
    """
    with open(fpath, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def inplace_change(filename, old_string, new_string):
    # Safely read the input filename using 'with'
    with open(filename) as f:
        s = f.read()
        if old_string not in s:
            print('"{old_string}" not found in {filename}.'.format(**locals()))
            return -1

    # Safely write the changed content, if found in the file
    with open(filename, 'w') as f:
        print('Inserting "{old_string}" to "{new_string}" in {filename}'.format(**locals()))
        s = s.replace(old_string, new_string)
        f.write(s)


def get_directory_structure(rootdir):
    """
    Creates a nested dictionary representing the folder structure of the specified root directory.

    Args:
        rootdir (str): The root directory to traverse.

    Returns:
        dict: A nested dictionary representing the folder structure of the root directory. The keys
              are the folder names and the values are dictionaries for subdirectories or None for files.

    Example:
        get_directory_structure('/home/user/documents')
        {
            'documents': {
                'folder1': {
                    'subfolder1': None,
                    'subfolder2': {
                        'file1.txt': None,
                        'file2.txt': None
                    },
                    'file3.txt': None
                },
                'folder2': {
                    'file4.txt': None
                }
            }
        }
    """
    dir = {}
    rootdir = os.path.join(rootdir)
    rootdir = rootdir.rstrip(os.sep)
    start = rootdir.rfind(os.sep) + 1
    for path, dirs, files in os.walk(rootdir):
        folders = path[start:].split(os.sep)
        subdir = dict.fromkeys(files)
        parent = reduce(dict.get, folders[:-1], dir)
        parent[folders[-1]] = subdir
    return dir


def get_filelist_fromdir(path):
    filelist = []
    for r, d, f in os.walk(path):
        for file in f:
            filelist.append(os.path.join(r, file))
    return filelist


def create_dirstructure(directories, path):
    """

    :param : directories - list of directories
    :param : path - path
    """
    for d in directories:
        Path(os.path.join(path, d)).mkdir(parents=True, exist_ok=True)
    return 0
