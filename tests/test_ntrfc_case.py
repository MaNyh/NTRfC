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


"""
def test_template_installations():
    import os
    from ntrfc.database.case_templates import CASE_TEMPLATES
    from ntrfc.utils.filehandling.datafiles import get_filelist_fromdir
    for name, template in CASE_TEMPLATES.items():
        assert os.path.isdir(template.path), "path to template does not exist"
        assert os.path.isfile(template.param_schema), "template-schema does not exist"
        assert len(get_filelist_fromdir(template.path)) > 0, "no files found in template"
"""
"""
def test_templates_params():
    from ntrfc.database.case_templates import CASE_TEMPLATES
    from ntrfc.utils.filehandling.datafiles import yaml_dict_read


    for name, template in CASE_TEMPLATES.items():
        paramschema = template.param_schema
        paramschema_dict = yaml_dict_read(paramschema)
        optionschema = template.option_schema
        optionschema_dict = yaml_dict_read(optionschema)
        default_params = {key: value["default"] for (key, value) in paramschema_dict["properties"].items()}
        default_options = {key: value["default"] for (key, value) in optionschema_dict["properties"].items()}

        template.set_params_options(default_params,default_options)

        assert template.sanity_check()
"""

"""
def test_create_case(tmpdir):
    import os
    from ntrfc.database.case_templates import CASE_TEMPLATES
    from ntrfc.utils.filehandling.datafiles import yaml_dict_read
    from ntrfc.database.case_creation import deploy

    template = list(CASE_TEMPLATES.values())[0]
    templatefiles = template.files
    templateparamschema = yaml_dict_read(template.param_schema)
    templateoptionschema = yaml_dict_read(template.option_schema)

    input = [f"{template.path}/{file}" for file in templatefiles]
    output = [f"{tmpdir}/{template.name}/{file}" for file in templatefiles]

    default_params = {key: value["default"] for (key, value) in templateparamschema["properties"].items()}
    default_options = {key: value["default"] for (key, value) in templateoptionschema["properties"].items()}

    deploy(input,output,default_params,default_options)
    check = [os.path.isfile(fpath) for fpath in output]
    assert all(check), "not all files have been created"
"""

#def test_find_variables_infile(tmpdir):
#    import os
#    from ntrfc.utils.filehandling.datafiles import get_directory_structure
#    from ntrfc.database.case_creation import find_variables_infile

#    paramnameone = "parameter_name_one"
#    paramnametwo = "parameter_name_two"

#    filecontent = f"""
#    <PARAM {paramnameone} PARAM>
#    <PARAM {paramnametwo} PARAM>
#    """
#    filename = "paramfile.txt"
#    with open(os.path.join(tmpdir, filename), "w") as fhandle:
#        fhandle.write(filecontent)

#    find_ans = find_variables_infile(os.path.join(tmpdir, filename),"PARAM")
#    assert list(find_ans.keys())[0]==paramnameone and list(find_ans.keys())[1]==paramnametwo

