import copy
import os
import re
import shutil
import warnings
from pathlib import Path

from ntrfc.utils.filehandling.datafiles import inplace_change, get_directory_structure
from ntrfc.utils.dictionaries.dict_utils import nested_dict_pairs_iterator, setInDict
from ntrfc.database.case_templates.templates import case_templates


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


def check_settings_necessarities(case_structure, settings_dict):

    necessarities = list(nested_dict_pairs_iterator(case_structure))
    necessarity_vars = []
    for item in necessarities:
        if item[-1] == "var":
            necessarity_vars.append(item[-2])

    defined_variables = list(settings_dict.keys())

    defined = []
    undefined = []
    unused = []
    used = []
    for variable in necessarity_vars:
        #assert variable in settings_variables, "variable " + variable + " not set in configuration file"
        if variable in defined_variables:
            defined.append(variable)
        else:
            undefined.append(variable)
    for variable in defined_variables:
        if not variable in necessarity_vars:
            unused.append(variable)
        else:
            used.append(variable)
    return defined, undefined, used, unused


def create_case(input, output, template, paras):
    """

    :param template: str - template-name
    :param settings: dict - dict-settings
    :param path_to_sim: path - path to case-directory
    :return:
    """
    #todo docstring and test method
    found = template in case_templates.keys()
    assert found, "template unknown. check ntrfc.database.casetemplates directory"
    TEMPLATEDIR=case_templates[template].path
    case_structure = get_directory_structure(os.path.join(TEMPLATEDIR, template))
    variables = find_vars_opts(case_structure, "var", list(nested_dict_pairs_iterator(case_structure)),os.path.join(TEMPLATEDIR))

    defined, undefined, used, unused = check_settings_necessarities(variables, paras)
    print("found ", str(len(defined)), " defined parameters")
    print("found ", str(len(undefined)), " undefined parameters")
    print("used ", str(len(used)), " parameters")
    print("unused ", str(len(unused)), " parameters")

    if len(undefined)>0:
        warnings.warn("undefined variables")
        return -1

    if len(unused)>0:
        warnings.warn("unused "+str(len(unused)))

    for templatefile, simfile in zip(input,output):
        shutil.copyfile(templatefile, simfile)
        for para in used:
            inplace_change(simfile,"<var " + para + " var>",str(paras[para]))


def find_vars_opts(case_structure, sign, all_pairs, path_to_sim):
    # todo docstring and test method
    # allowing names like JOB_NUMBERS, only capital letters and underlines - no digits, no whitespaces
    datadict = copy.deepcopy(case_structure)
    varsignature = r"<PLACEHOLDER [A-Z]{3,}(_{1,1}[A-Z]{3,}){,} PLACEHOLDER>".replace(r'PLACEHOLDER', sign)
    #int
    #float
    #string
    siglim = (len(sign)+2, -(len(sign)+2))

    for pair in all_pairs:
        #if os.path.isfile(os.path.join(path_to_sim,*pair)):
        setInDict(datadict, pair[:-1], {})
        filepath = os.path.join(*pair[:-1])
        with open(os.path.join(path_to_sim, filepath), "r") as fhandle:
            for line in fhandle.readlines():
                datadict = search_paras(datadict, line, pair, siglim, varsignature, sign)
    return datadict