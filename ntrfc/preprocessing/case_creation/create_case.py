import copy
import os
import re
import shutil
from pathlib import Path

from ntrfc.utils.filehandling.datafiles import inplace_change, get_directory_structure
from ntrfc.utils.dictionaries.dict_utils import nested_dict_pairs_iterator, setInDict,merge
from database.case_templates.case_templates import CASE_TEMPLATES


def create_simdirstructure(case_structure, path):
    # todo docstring and test method
    directories = list(nested_dict_pairs_iterator(case_structure))
    for d in directories:
        dirstructure = d[:-2]
        Path(os.path.join(path, *dirstructure)).mkdir(parents=True, exist_ok=True)
    return 0


def search_paras(case_structure, line, pair, siglim, varsignature):
    """

    """
    # todo docstring and test method
    lookforvar = True
    while (lookforvar):
        lookup_var = re.search(varsignature, line)
        if not lookup_var:
            lookforvar = False
        else:
            span = lookup_var.span()
            parameter = line[span[0] + siglim[0]:span[1] + siglim[1]]
            setInDict(case_structure, list(pair[:-1]) + [parameter])
            match = line[span[0]:span[1]]
            line = line.replace(match, "")
    return case_structure


def check_settings_necessarities(case_structure, settings_dict):
    """

    : params: case_structure dict
    """

    necessarities = list(nested_dict_pairs_iterator(case_structure))
    necessarity_vars = []
    for item in necessarities:
        if item[-1] in PARAMS.keys():
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


def test_create_case(tmpdir):
    from ntrfc.database.case_templates.case_templates import create_filelist_from_template
    template = list(CASE_TEMPLATES.values())[0]
    TEMPLATEFILES = template.files

    input = [f"{TEMPLATE_PATH}/{file}" for file in TEMPLATEFILES]
    output = [f"{tmpdir}/{file}" for file in TEMPLATEFILES]
    paras = {}
    create_case(input,output,template,paras)

def create_case(input, output, template, paras):
    """

    :param template: str - template-name
    :param settings: dict - dict-settings
    :param path_to_sim: path - path to case-directory
    :return:
    """

    found = template in CASE_TEMPLATES.keys()
    assert found, "template unknown. check ntrfc.database.casetemplates directory"
    TEMPLATEDIR = CASE_TEMPLATES[template].path
    case_structure = get_directory_structure(os.path.join(TEMPLATEDIR, template))

    variables=find_vars_opts(case_structure, list(nested_dict_pairs_iterator(case_structure)), TEMPLATEDIR)

    defined, undefined, used, unused = check_settings_necessarities(variables, paras)
    print("found ", str(len(defined)), " defined parameters")
    print("found ", str(len(undefined)), " undefined parameters")
    print("used ", str(len(used)), " parameters")
    print("unused ", str(len(unused)), " parameters")

    assert len(undefined)==0, "undefined parameters"

    for templatefile, simfile in zip(input,output):
        shutil.copyfile(templatefile, simfile)
        for parameter in used:
            inplace_change(simfile,f"<var {parameter} var>",str(paras[parameter]))


def find_vars_opts(case_structure,all_pairs, path_to_sim):
    """
    : param case_structure: dict - case-structure. can carry parameters
    : param sign: str - sign of a parameter (Velocity -> U etc.)
    : param all_pairs: dict - ?
    : param path_to_sim: path - path-like object
    return : ?
    """
    # todo docstring and test method
    # allowing names like JOB_NUMBERS, only capital letters and underlines - no digits, no whitespaces
    datadict = copy.deepcopy(case_structure)
    varsignature = r"<PARAM [a-z]{3,}(_{1,1}[a-z]{3,}){,} PARAM>"
    #int
    #float
    #string
    siglim = (len("<PARAM "), -(len(" PARAM")+2))

    for pair in all_pairs:
        #if os.path.isfile(os.path.join(path_to_sim,*pair)):
        setInDict(datadict, pair[:-1], {})
        filepath = os.path.join(*pair[:-1])
        with open(os.path.join(path_to_sim, filepath), "r") as fhandle:
            for line in fhandle.readlines():
                datadict = merge(search_paras(datadict, line, pair, siglim, varsignature),datadict)
    return datadict
