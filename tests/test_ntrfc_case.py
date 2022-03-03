#!/usr/bin/env python

"""Tests for `ntrfc` package."""

def test_template_installations():
    """
    basic sanity check over the installed templates
    """
    import os
    from database.case_templates.case_templates import CASE_TEMPLATES
    from database.case_templates.case_templates import create_filelist_from_template
    for name, template in CASE_TEMPLATES.items():
        assert os.path.isdir(template.path), "path to template does not exist"
        assert os.path.isfile(template.schema), "template-schema does not exist"
        assert len(create_filelist_from_template(name))>0, "no files found in template"

def test_templates():
    """

    """
    import os
    from database.case_templates.case_templates import CASE_TEMPLATES
    from ntrfc.utils.dictionaries.dict_utils import merge
    from ntrfc.utils.filehandling.datafiles import yaml_dict_read
    from ntrfc.utils.filehandling.datafiles import get_directory_structure
    from ntrfc.preprocessing.case_creation.create_case import find_vars_opts,check_settings_necessarities
    from ntrfc.utils.dictionaries.dict_utils import nested_dict_pairs_iterator
    from ntrfc.database.parameters.parameter import PARAMS


    for name, template in CASE_TEMPLATES.items():
        schema = template.schema
        schema_dict=yaml_dict_read(schema)
        default_params={name:{key: value["default"] for (key, value) in schema_dict["properties"].items()}}
        path = template.path
        tpath = os.path.join(path,"..")
        case_structure = get_directory_structure(path)
        casefiles = list(nested_dict_pairs_iterator(case_structure))
        variables = {}
        for param, paramkwargs in PARAMS.items():
            variables = merge(
                find_vars_opts(case_structure, param, casefiles, tpath), variables)


        defined, undefined, used, unused= check_settings_necessarities(variables,default_params[name])
        assert len(undefined)==0, "some parameters have no default"
        assert len(unused)==0, "some parameters are not used"
