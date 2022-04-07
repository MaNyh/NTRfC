from ntrfc.database.case_creation import case_template

"""
avail_templates is used to create a dictionary of with case_templates-Objects
listed templates in avail_templates will be checked via tests
only listed templates in avail_templates can be called via a workflow
"""

AVAIL_TEMPLATES = ["openfoamCompressorCascadeRas"]
# create dict of case-templates from AVAIL_TEMPLATES
CASE_TEMPLATES = {templatename: case_template(templatename) for templatename in AVAIL_TEMPLATES}
