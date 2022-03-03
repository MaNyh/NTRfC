import os
import importlib_resources

from ntrfc.utils.dictionaries.dict_utils import nested_dict_pairs_iterator
from ntrfc.utils.filehandling.datafiles import get_directory_structure

class case_template:
    def __init__(self,name):
        self.name = name
        self.path = importlib_resources.files("ntrfc") / f"database/case_templates/{name}"
        self.schema = importlib_resources.files("ntrfc") / f"database/case_templates/{name}.schema.yaml"

"""
avail_templates is used to create a dictionary of with case_templates-Objects
listed templates in avail_templates will be checked via tests
only listed templates in avail_templates can be called via a workflow
"""

AVAIL_TEMPLATES = ["openfoamCompressorCascadeRas"]
#create dict of case-templates from AVAIL_TEMPLATES
CASE_TEMPLATES = {templatename: case_template(templatename) for templatename in AVAIL_TEMPLATES}


def create_filelist_from_template(templatename):
    """
    :param template: path
    :return: list of files
    """
    templatepath = CASE_TEMPLATES[templatename].path
    assert os.path.isdir(os.path.join(templatepath))
    files = list(nested_dict_pairs_iterator(get_directory_structure(templatepath)))
    fpaths = [os.path.join(*i[1:-1]) for i in files]
    return fpaths
