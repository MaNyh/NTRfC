from snakemake.utils import validate
from snakemake.utils import Paramspace
import pandas as pd

from ntrfc.database.case_templates.case_templates import CASE_TEMPLATES

configfile : "config/casesettings.yaml"

def validate_configuration(config):
    validate(config,"../schemas/config.schema.yaml")
    template = CASE_TEMPLATES[config["case_params"]["case_type"]]
    PARAMS = pd.read_csv("config/caseparams.tsv",sep="\t")
    validate(PARAMS, template.schema)
    paramspace = Paramspace(PARAMS)
    return template, paramspace, config

template, paramspace, config = validate_configuration(config)

def get_casefiles():
    return [f"results/simulations/{instance_pattern}/{file}" for instance_pattern in paramspace.instance_patterns for file
      in template.files]



rule create_case:
    input:
        [f"{template.path}/{file}" for file in template.files]
    output:
        # format a wildcard pattern like "alpha~{alpha}/beta~{beta}/gamma~{gamma}"
        # into a file path, with alpha, beta, gamma being the columns of the data frame
        *[f"results/simulations/{paramspace.wildcard_pattern}/{file}" for file in template.files]
    params:
        # automatically translate the wildcard values into an instance of the param space
        # in the form of a dict (here: {"alpha": ..., "beta": ..., "gamma": ...})
        simparams = paramspace.instance
    run:
        from ntrfc.database.case_templates.case_creation import deploy

        deploy(input,output,params["simparams"],config["case_options"])
