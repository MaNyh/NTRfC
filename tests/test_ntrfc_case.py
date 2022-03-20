#!/usr/bin/env python

"""Tests for `ntrfc` package."""


def test_casestructure(tmpdir):
    import os
    from ntrfc.utils.filehandling.datafiles import get_directory_structure, create_dirstructure
    from ntrfc.utils.dictionaries.dict_utils import nested_dict_pairs_iterator

    test_structure = {"directory": {"dir1_file1": "",
                                    "dir1_file2": "",
                                    }}

    directories = [os.path.join(*i[:-1]) for i in list(nested_dict_pairs_iterator(test_structure))]

    create_dirstructure(directories, tmpdir)

    case_structure = get_directory_structure(tmpdir)

    assert test_structure["directory"].keys() == case_structure[tmpdir.basename][
        "directory"].keys(), "error collecting case_structure"


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
    from ntrfc.database.case_templates.case_templates import CASE_TEMPLATES
    from ntrfc.utils.filehandling.datafiles import yaml_dict_read


    for name, template in CASE_TEMPLATES.items():
        schema = template.schema
        schema_dict = yaml_dict_read(schema)
        default_params = {key: value["default"] for (key, value) in schema_dict["properties"].items()}

        template.set_params_options(default_params,options)
        assert len(undefined) == 0, f"some parameters have no default: {undefined}"
        assert len(unused) == 0, f"some parameters are not used: {unused}"


def test_create_case(tmpdir):
    """

    """
    import os
    from ntrfc.database.case_templates.case_templates import CASE_TEMPLATES, deploy
    from ntrfc.utils.filehandling.datafiles import yaml_dict_read
    from ntrfc.utils.filehandling.datafiles import create_dirstructure

    template = list(CASE_TEMPLATES.values())[0]
    templatefiles = template.files
    templateschema = yaml_dict_read(template.schema)
    directories = [os.path.dirname(fpath) for fpath in templatefiles]

    input = [f"{template.path}/{file}" for file in templatefiles]
    output = [f"{tmpdir}/{template.name}/{file}" for file in templatefiles]

    default_params = {key: value["default"] for (key, value) in templateschema["properties"].items()}

    #sim_params = merge(default_params, default_options)

    os.mkdir(os.path.join(tmpdir, template.name))
    create_dirstructure(directories, os.path.join(tmpdir, template.name))

    create_case(input, output, template.name, default_params)
    check = [os.path.isfile(fpath) for fpath in output]
    assert all(check), "not all files have been created"


def test_search_paras(tmpdir):
    import os
    from ntrfc.utils.filehandling.datafiles import get_directory_structure
    from ntrfc.utils.dictionaries.dict_utils import nested_dict_pairs_iterator

    paramnameone = "parameter_name_one"
    paramnametwo = "parameter_name_two"

    filecontent = f"""
    <PARAM {paramnameone} PARAM>
    <PARAM {paramnametwo} PARAM>
    """
    filename = "paramfile.txt"
    with open(os.path.join(tmpdir, filename), "w") as fhandle:
        fhandle.write(filecontent)
    case_structure = get_directory_structure(tmpdir)

    varsignature = r"<PARAM [a-z]{3,}(_{1,1}[a-z]{3,}){,} PARAM>"
    all_pairs = list(nested_dict_pairs_iterator(case_structure))
    for line in filecontent:
        for pair in all_pairs:
            search_paras(case_structure, line, pair, (len("<PARAM "), len(" PARAM>")), varsignature,"PARAM")
