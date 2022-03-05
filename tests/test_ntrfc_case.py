#!/usr/bin/env python

"""Tests for `ntrfc` package."""
import os

from database.case_templates.case_templates import CASE_TEMPLATES
from preprocessing.case_creation.create_case import create_case


def test_template_installations():
    """
    basic sanity check over the installed templates
    """
    import os
    from ntrfc.database.case_templates.case_templates import CASE_TEMPLATES
    from ntrfc.utils.filehandling.datafiles import get_filelist_fromdir
    for name, template in CASE_TEMPLATES.items():
        assert os.path.isdir(template.path), "path to template does not exist"
        assert os.path.isfile(template.schema), "template-schema does not exist"
        assert len(get_filelist_fromdir(template.path)) > 0, "no files found in template"


def test_templates_params():
    """

    """
    import os
    from ntrfc.database.case_templates.case_templates import CASE_TEMPLATES
    from ntrfc.utils.filehandling.datafiles import yaml_dict_read
    from ntrfc.utils.filehandling.datafiles import get_directory_structure
    from ntrfc.preprocessing.case_creation.create_case import find_vars_opts, check_settings_necessarities

    for name, template in CASE_TEMPLATES.items():
        schema = template.schema
        schema_dict = yaml_dict_read(schema)
        default_params = {name: {key: value["default"] for (key, value) in schema_dict["properties"].items()}}
        path = template.path
        tpath = os.path.join(path, "..")
        case_structure = get_directory_structure(path)
        variables = find_vars_opts(case_structure, tpath)
        defined, undefined, used, unused = check_settings_necessarities(variables, default_params[name])
        assert len(undefined) == 0, f"some parameters have no default: {undefined}"
        assert len(unused) == 0, f"some parameters are not used: {unused}"


def test_create_case(tmpdir):
    """

    """
    from ntrfc.utils.filehandling.datafiles import yaml_dict_read
    from ntrfc.utils.filehandling.datafiles import create_dirstructure


    template = list(CASE_TEMPLATES.values())[0]
    templatefiles = template.files
    templateschema = yaml_dict_read(template.schema)
    directories = [os.path.dirname(fpath) for fpath in templatefiles]

    input = [f"{template.path}/{file}" for file in templatefiles]
    output = [f"{tmpdir}/{template.name}/{file}" for file in templatefiles]
    paras = {k:v["default"] for k,v in templateschema["properties"].items()}
    os.mkdir(os.path.join(tmpdir,template.name))
    create_dirstructure(directories,os.path.join(tmpdir,template.name))
    create_case(input,output,template.name,paras)
    check = [os.path.isfile(fpath) for fpath in output]
    assert all(check), "not all files have been created"
