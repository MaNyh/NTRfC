import os
import importlib_resources

from ntrfc.utils.dictionaries.dict_utils import nested_dict_pairs_iterator
from ntrfc.utils.filehandling.datafiles import get_directory_structure


class case_template:
    def __init__(self,name):
        self.name = name
        self.path = importlib_resources.files("ntrfc") / f"database/case_templates/{name}"
        self.schema = importlib_resources.files("ntrfc") / f"database/case_templates/{name}.schema.yaml"


avail_templates = ["traceCompressorCascadeRas"]

case_templates = {}
for templatename in avail_templates:
    case_templates[templatename] = case_template("traceCompressorCascadeRas")


def create_filelist_from_template(templatename):
    """
    :param template: path
    :return: list of files
    """
    templatepath = case_templates[templatename].path
    assert os.path.isdir(os.path.join(templatepath))
    files = list(nested_dict_pairs_iterator(get_directory_structure(templatepath)))
    fpaths = [os.path.join(*i[1:-1]) for i in files]
    return fpaths
