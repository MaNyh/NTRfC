from snakemake.utils import validate
from snakemake.utils import Paramspace
from snakemake import load_configfile

import pandas as pd

from ntrfc.database.case_templates import CASE_TEMPLATES


template = CASE_TEMPLATES[config["case_params"]["case_type"]]

params = pd.read_csv("config/case_params.tsv",sep="\t")
validate(params, template.param_schema)
paramspace = Paramspace(params)

option_config = load_configfile("config/case_options.yaml")
validate(option_config,template.option_schema)


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
        from ntrfc.database.case_creation import deploy

        deploy(input,output,params["simparams"],option_config)
