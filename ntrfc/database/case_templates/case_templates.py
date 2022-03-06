import importlib_resources
import os

from ntrfc.utils.filehandling.datafiles import get_filelist_fromdir


class case_template:
    def __init__(self, name):
        self.name = name
        self.path = importlib_resources.files("ntrfc") / f"database/case_templates/{name}"
        self.schema = importlib_resources.files("ntrfc") / f"database/case_templates/{name}.schema.yaml"
        self.files = [os.path.relpath(fpath, self.path) for fpath in get_filelist_fromdir(self.path)]


"""
avail_templates is used to create a dictionary of with case_templates-Objects
listed templates in avail_templates will be checked via tests
only listed templates in avail_templates can be called via a workflow
"""

AVAIL_TEMPLATES = ["openfoamCompressorCascadeRas"]
# create dict of case-templates from AVAIL_TEMPLATES
CASE_TEMPLATES = {templatename: case_template(templatename) for templatename in AVAIL_TEMPLATES}
