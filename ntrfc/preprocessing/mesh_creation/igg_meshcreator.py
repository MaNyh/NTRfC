import os
import csv

meshdir = "04_mesh"

if not os.path.isdir(meshdir):
    os.mkdir(meshdir)

os.chdir(meshdir)

with open("mesh_config.csv", 'r') as f:
    mesh_configreader = csv.reader(f, delimiter=' ')
    mesh_config = {}
    for line in mesh_configreader:
        mesh_config[line[0]] = line[1]


print("hey")
print(os.getcwd())
print(mesh_config)
print("ney")

save_project("mesh.igg")
export_FLUENT("fluent.msh")

os.chdir("..")
