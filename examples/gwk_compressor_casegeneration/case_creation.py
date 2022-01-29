import os
from functools import reduce
import shutil
from pathlib import Path
import copy
import re

from ntrfc.utils.dictionaries.dictutils import nested_dict_pairs_iterator, setInDict

TEMPLATEDIR = r"D:\CodingProjects\NTRfC\ntrfc\database\case_templates"
path_to_sim=r"D:\CodingProjects\NTRfC\examples\gwk_compressor_casegeneration\01_case"
TEMPLATES = [i for i in os.listdir(TEMPLATEDIR) if os.path.isdir(os.path.join(TEMPLATEDIR, i))]

def get_directory_structure(rootdir):
    """
    Creates a nested dictionary that represents the folder structure of rootdir
    """
    #test method
    dir = {}
    rootdir = rootdir.rstrip(os.sep)
    start = rootdir.rfind(os.sep) + 1
    for path, dirs, files in os.walk(rootdir):
        folders = path[start:].split(os.sep)
        subdir = dict.fromkeys(files)
        parent = reduce(dict.get, folders[:-1], dir)
        parent[folders[-1]] = subdir
    return dir



def find_vars_opts(case_structure, sign, all_pairs, path_to_sim):
    # todo docstring and test method
    # allowing names like JOB_NUMBERS, only capital letters and underlines - no digits, no whitespaces
    datadict = copy.deepcopy(case_structure)
    varsignature = r"<PLACEHOLDER [A-Z]{3,}(_{1,1}[A-Z]{3,}){,} PLACEHOLDER>".replace(r'PLACEHOLDER', sign)
    siglim = (len(sign)+2, -(len(sign)+2))

    for pair in all_pairs:
        #if os.path.isfile(os.path.join(path_to_sim,*pair)):
        setInDict(datadict, pair[:-1], {})
        filepath = os.path.join(*pair[:-1])
        with open(os.path.join(path_to_sim, filepath), "r") as fhandle:
            for line in fhandle.readlines():
                datadict = search_paras(datadict, line, pair, siglim, varsignature, sign)
    return datadict

def create_simdirstructure(case_structure, path):
    # todo docstring and test method
    directories = list(nested_dict_pairs_iterator(case_structure))
    for d in directories:
        dirstructure = d[:-2]
        Path(os.path.join(path,*dirstructure)).mkdir(parents=True, exist_ok=True)
    return 0


def search_paras(case_structure, line, pair, siglim, varsignature, varsign):
    # todo docstring and test method
    lookforvar = True
    while (lookforvar):
        lookup_var = re.search(varsignature, line)
        if not lookup_var:
            lookforvar = False
        else:
            span = lookup_var.span()
            parameter = line[span[0] + siglim[0]:span[1] + siglim[1]]
            setInDict(case_structure, list(pair[:-1]) + [parameter], varsign)
            match = line[span[0]:span[1]]
            line = line.replace(match, "")
    return case_structure

def writeout_simulation(case_structure_parameters, path_to_sim, settings):
    walk_casefile_list = nested_dict_pairs_iterator(case_structure_parameters)
    for parameterdata in walk_casefile_list:
        fpath = os.path.join(path_to_sim, *parameterdata[:-2])
        parametername = parameterdata[-2]

        para_type = parameterdata[-1]
        if para_type == "var":
            para_type = "variables"
            variable = settings["simcase_settings"][para_type][parametername]
            with open(fpath) as fobj:
                newText = fobj.read().replace("<var " + parametername + " var>", str(variable))
            with open(fpath, "w") as fobj:
                fobj.write(newText)

"""def check_settings_necessarities(case_structure, settings_dict):

    #todo not used yet, might be handy!
    necessarities = list(nested_dict_pairs_iterator(case_structure))
    necessarity_vars = []
    for item in necessarities:
        if item[-1] == "var":
            necessarity_vars.append(item[-2])

    assert "variables" in settings_dict["simcase_settings"].keys(), "variables not set simcase_settings"
    if settings_dict["simcase_settings"]["variables"]:
        settings_variables = list(settings_dict["simcase_settings"]["variables"].keys())
    else:
        settings_variables = []

    for variable in necessarity_vars:
        assert variable in settings_variables, "variable " + variable + " not set in configuration file"
"""

def create_case_fromtemplate(template, settings, path_to_sim):
    #todo docstring and test method
    found = template in TEMPLATES
    assert found, "template unknown. check ntrfc.database.casetemplates directory"

    case_structure = get_directory_structure(os.path.join(TEMPLATEDIR, template))
    case_files = [i[:-1] for i in list(nested_dict_pairs_iterator(case_structure)) if os.path.isfile(os.path.join(TEMPLATEDIR,*list(i[:-1])))]

    for fpath in case_files:
        filename = fpath[-1]
        dirstructure = fpath[:-1]
        if dirstructure == ():
            dirstructure = ""

        template_fpath = os.path.join(TEMPLATEDIR,
                                      *dirstructure,
                                      filename)
        create_simdirstructure(case_structure,path_to_sim)
        sim_fpath = os.path.join(path_to_sim, *dirstructure)

        shutil.copyfile(template_fpath, os.path.join(sim_fpath,filename))
    variables = find_vars_opts(case_structure, "var", list(nested_dict_pairs_iterator(case_structure)),"01_case")
    nov = len(list(nested_dict_pairs_iterator(variables)))
    print("found ", str(nov), "parameters of type var in copied template")
    #check_settings_necessarities(variables, settings)
    return case_files

settings = {}
case_structure = create_case_fromtemplate('trace-compressor-cascade-ras', settings, path_to_sim)
