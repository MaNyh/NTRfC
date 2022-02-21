from snakemake.utils import validate


configfile: "NTRfC/ntrfc/config/config.yaml"
validate(config, "NTRfC/ntrfc/config/config.schema.yaml")


rule mesh:
    output:
        r"meshing/mesh1.txt", r"meshing/mesh2.txt"
    shell:
        "touch {output}"

