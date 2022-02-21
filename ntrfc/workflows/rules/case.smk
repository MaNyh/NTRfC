include: "mesh.smk"


rule Case:
    input: rules.mesh.output

    params:
        mycase= config["case"]["my_case"]

    output: r"Case/case.txt"
    shell:
        "touch {output}"

