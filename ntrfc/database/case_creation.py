import os
import re
import shutil
import warnings
from functools import reduce

import importlib_resources

from ntrfc.utils.dictionaries.dict_utils import nested_dict_pairs_iterator
from ntrfc.utils.filehandling.datafiles import inplace_change, get_filelist_fromdir


def get_directory_structure(rootdir):
    """
    Creates a nested dictionary that represents the folder structure of rootdir
    """
    # test method
    dir = {}
    rootdir = os.path.join(rootdir)
    rootdir = rootdir.rstrip(os.sep)
    start = rootdir.rfind(os.sep) + 1
    for path, dirs, files in os.walk(rootdir):
        folders = path[start:].split(os.sep)
        subdir = dict.fromkeys(files)
        parent = reduce(dict.get, folders[:-1], dir)
        parent[folders[-1]] = subdir
    return dir[os.path.basename(rootdir)]


def find_vars(path_to_sim, sign):
    """
    : param case_structure: dict - case-structure. can carry parameters
    : param sign: str - sign of a parameter (Velocity -> U etc.)
    : param all_pairs: dict - ?
    : param path_to_sim: path - path-like object
    return : ?
    """
    case_structure = get_directory_structure(path_to_sim)
    all_files = [i[:-1] for i in list(nested_dict_pairs_iterator(case_structure))]

    sim_variables = {}
    for file in all_files:
        filepath = os.path.join(path_to_sim, *file)
        filevars =find_variables_infile(filepath, sign)
        for k, v in filevars.items():
            if k not in sim_variables:
                sim_variables[k]=v
            else:
                sim_variables[k].append(v)
    return sim_variables


def find_variables_infile(file, sign):

    varsignature = r"<PLACEHOLDER [a-z]{3,}(_{1,1}[a-z]{3,}){,} PLACEHOLDER>".replace("PLACEHOLDER", sign)
    siglim = (len(f"< {sign}"), -(len(f" {sign}>")))
    variables = {}
    with open(file, "r") as fhandle:
        for line in fhandle.readlines():
            lookaround = True
            while lookaround:
                lookup_var = re.search(varsignature, line)
                if not lookup_var:
                    lookaround = False
                    assert sign not in line, f"parameter is not defined correct \n file: {filepath}\n line: {line}"
                else:
                    span = lookup_var.span()
                    parameter = line[span[0] + siglim[0]:span[1] + siglim[1]]
                    # update
                    if parameter not in variables.keys():
                        variables[parameter] = []
                    if file not in variables[parameter]:
                        variables[parameter].append(file)
                    match = line[span[0]:span[1]]
                    line = line.replace(match, "")
    return variables


def deploy(deply_sources,deploy_targets, deploy_params, deploy_options):
    for source, target in zip(deply_sources,deploy_targets):
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copyfile(source, target)
        for parameter in deploy_params:
            inplace_change(target, f"<PARAM {parameter} PARAM>", str(deploy_params[parameter]))
        for option in deploy_options:
            inplace_change(target, f"<OPTION {option} OPTION>", str(deploy_options[option]))


class case_template:

    psign = "PARAM"
    osign = "OPTION"

    def __init__(self, name):
        self.name = name
        self.path = importlib_resources.files("ntrfc") / f"../cases/{name}"
        self.param_schema = importlib_resources.files("ntrfc") / f"../cases/{name}_param.schema.yaml"
        self.option_schema = importlib_resources.files("ntrfc") / f"../cases/{name}_option.schema.yaml"
        self.files = [os.path.relpath(fpath, self.path) for fpath in get_filelist_fromdir(self.path)]

        self.params = find_vars(self.path, self.psign)
        self.params_set = {}
        self.options = find_vars(self.path,self.osign)
        self.options_set = {}

    def set_params_options(self,params_set,options_set):
        self.params_set = params_set
        self.options_set = options_set

    def sanity_check(self):
        sanity = True
        for p in self.params.keys():
            if p not in self.params_set.keys():
                sanity=False
                warnings.warn(f"{p} not set")
        for o in self.options.keys():
            if o not in self.options_set.keys():
                sanity=False
                warnings.warn(f"{o} not set")
        return sanity

class dynamic_case_template(case_template):
    def __init__(self,name,path,schema):
        super.__init__(name)
        self.path = path
        self.schema = schema
        self.files = [os.path.relpath(fpath, self.path) for fpath in get_filelist_fromdir(self.path)]

        self.params = find_vars(self.path, self.psign)
        self.params_set = {}
        self.options = find_vars(self.path,self.osign)
        self.options_set = {}
