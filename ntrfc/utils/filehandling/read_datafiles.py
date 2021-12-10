import yaml
import pickle

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

def write_yaml_dict(fpath,data):
    """
    :param fpath: target path
    :param data: dictionary
    """
    with open(fpath, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

