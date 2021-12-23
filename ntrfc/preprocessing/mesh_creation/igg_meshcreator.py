import os
import csv

meshdir = "04_mesh"

if not os.path.isdir(meshdir):
    os.mkdir(meshdir)

os.path.chdir(meshdir)

with open("mesh_config.csv", 'r') as f:
    mesh_configreader = csv.reader(f, delimiter=' ')
    mesh_config = {}
    for line in mesh_configreader:
        mesh_config[line[0]] = line[1]


print("hey")
print(mesh_config)
print("ney")

save_project("04_mesh/mesh.igg")
export_FLUENT("04_mesh/fluent.msh")

os.path.chdir("..")
