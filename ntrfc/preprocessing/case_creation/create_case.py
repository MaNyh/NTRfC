import copy
import os
import re
import shutil

from ntrfc.utils.filehandling.datafiles import inplace_change, get_directory_structure
from ntrfc.utils.dictionaries.dict_utils import nested_dict_pairs_iterator, set_in_dict, merge
from ntrfc.database.case_templates.case_templates import CASE_TEMPLATES


def search_paras(case_structure, line, pair, siglim, varsignature, sign):
    """

    """
    # todo docstring and test method
    lookforvar = True
    while (lookforvar):
        lookup_var = re.search(varsignature, line)
        if not lookup_var:
            lookforvar = False
            filename = os.path.join(*pair[:-1])
            assert sign not in line, f"parameter is not defined correct \n file: {filename}\n line: {line}"
        else:
            span = lookup_var.span()
            parameter = line[span[0] + siglim[0]:span[1] + siglim[1]]
            # update
            set_in_dict(case_structure, list(pair[:-1]) + [parameter], sign)
            match = line[span[0]:span[1]]
            line = line.replace(match, "")
    return case_structure


def settings_sanity(case_structure, settings_dict):
    """

    : params: case_structure dict
    """

    necessarities = list(nested_dict_pairs_iterator(case_structure))
    necessarity_vars = []
    for item in necessarities:
        if item[-1] == "PARAM" or item[-1] == "OPTION":
            necessarity_vars.append(item[-2])

    defined_variables = list(settings_dict.keys())

    defined = []
    undefined = []
    unused = []
    used = []
    for variable in necessarity_vars:
        if variable in defined_variables:
            defined.append(variable)
        else:
            undefined.append(variable)
    for variable in defined_variables:
        if variable not in necessarity_vars:
            unused.append(variable)
        else:
            used.append(variable)
    return defined, undefined, used, unused


def create_case(input_files, output_files, template_name, simparams):
    """
    :param input_files: list of template-files
    :param output_files: list of outputfiles (same as input)
    :param template_name: str - template-name
    :param simparams: dict - dict-settings - passed via filenames
    :return:
    """

    found = template_name in CASE_TEMPLATES.keys()
    assert found, "template unknown. check ntrfc.database.casetemplates directory"
    template = CASE_TEMPLATES[template_name]
    case_structure = get_directory_structure(template.path)

    param_sign = "PARAM"
    option_sign = "OPTION"

    parameters = find_vars_opts(case_structure[template.name], template.path, param_sign)
    options = find_vars_opts(case_structure[template.name], template.path, option_sign)
    case_settings = merge(parameters, options)

    allparams = merge(parameters,options)
    defined, undefined, used, unused = settings_sanity(case_settings, simparams)
    print("found ", str(len(defined)), " defined parameters")
    print("found ", str(len(undefined)), " undefined parameters")
    print("used ", str(len(used)), " parameters")
    print("unused ", str(len(unused)), " parameters")

    assert len(undefined) == 0, f"undefined parameters: {undefined}"
    assert len(unused) == 0, f"unused parameters: {unused}"

    necessarities = list(nested_dict_pairs_iterator(case_settings))
    paramtypes = {}
    for item in necessarities:
        if item[-1] == "PARAM":
            paramtypes[item[-2]]="PARAM"
        elif item[-1] == "OPTION":
            paramtypes[item[-2]] = "OPTION"

    for templatefile, simfile in zip(input_files, output_files):
        shutil.copyfile(templatefile, simfile)
        for parameter in used:
            sign = paramtypes[parameter]
            inplace_change(simfile, f"<{sign} {parameter} {sign}>", str(simparams[parameter]))


def find_vars_opts(case_structure, path_to_sim, sign):
    """
    : param case_structure: dict - case-structure. can carry parameters
    : param sign: str - sign of a parameter (Velocity -> U etc.)
    : param all_pairs: dict - ?
    : param path_to_sim: path - path-like object
    return : ?
    """
    # allowing names like JOB_NUMBERS, only capital letters and underlines - no digits, no whitespaces
    datadict = copy.deepcopy(case_structure)
    all_pairs = list(nested_dict_pairs_iterator(case_structure))
    varsignature = r"<PLACEHOLDER [a-z]{3,}(_{1,1}[a-z]{3,}){,} PLACEHOLDER>".replace("PLACEHOLDER", sign)
    # int
    # float
    # string
    # todo move into param-module
    siglim = (len(f"< {sign}"), -(len(f" {sign}>")))

    for pair in all_pairs:
        # if os.path.isfile(os.path.join(path_to_sim,*pair)):
        set_in_dict(datadict, pair[:-1], {})
        filepath = os.path.join(*pair[:-1])
        with open(os.path.join(path_to_sim, filepath), "r") as fhandle:
            for line in fhandle.readlines():
                datadict = search_paras(datadict, line, pair, siglim, varsignature, sign)
    return datadict
