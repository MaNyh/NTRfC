import snakemake

c = snakemake.load_configfile("casesettings.yaml")
snakemake.utils.validate(c,'config.schema.yaml')

snakemake.snakemake(snakefile="Snakefile",)
