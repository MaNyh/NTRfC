include: "run.smk"

rule Post:
    input: rules.Run.output

    params:
        rs= config["post_processing"]["param"]

    output: r"Post/results.txt"
    shell:
        "touch {output}"

