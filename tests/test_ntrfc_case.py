#!/usr/bin/env python

"""Tests for `ntrfc` package."""

def test_templates():
    """
    basic sanity check over the installed templates
    """
    import os
    from ntrfc.database.case_templates.templates import case_templates
    from ntrfc.database.case_templates.templates import create_filelist_from_template
    for name, template in case_templates.items():
        assert os.path.isdir(template.path), "path to template does not exist"
        assert os.path.isfile(template.schema), "template-schema does not exist"
        assert len(create_filelist_from_template(name))>0, "no files found in template"
