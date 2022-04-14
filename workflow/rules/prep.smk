import os
import shutil
from ntrfc.utils.filehandling.datafiles import get_filelist_fromdir

def get_constantlist_fromdir(path):
    dirlist = []
    for r, d, f in os.walk(path):
        for folder in d:
            if folder=="constant":
                dirlist.append(os.path.join(r, folder))
    return dirlist

def get_Simulationdir(path):
    dirlist = []
    for r, d, f in os.walk(path):
        for folder in d:
            if folder=="pitch~0.0765":
                dirlist.append(os.path.join(r, folder))
    return dirlist

script = """
module load GCC/11.2.0  OpenMPI/4.1.1 OpenFOAM/v2112;
set +u;
source $FOAM_BASH;
set -u;
fluent3DMeshToFoam fluent.msh;
createPatch -overwrite;
topoSet;
decomposePar;
"""
rule prep:
    input: "fluent.msh", get_filelist_fromdir("results/simulations")
    output: directory([f"{dir}/polyMesh" for dir in get_constantlist_fromdir("results/simulations")]),
            directory([f"{dir}/polyMesh/sets" for dir in get_constantlist_fromdir("results/simulations")])
    run:
        for pth in get_Simulationdir("results/simulations"):
            snakepth = os.getcwd()
            os.chdir(pth)
            new = os.getcwd()
            shutil.copy(snakepth +"/"+ "fluent.msh", new + "/"+ "fluent.msh")
            os.system("bash -c '%s'" % script)
            os.remove("fluent.msh")
            os.chdir(snakepth)

