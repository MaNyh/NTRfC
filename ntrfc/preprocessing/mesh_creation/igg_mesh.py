from ntrfc.utils.filehandling.datafiles import write_csv_config
import os

def run_igg(basedir,profile_name_sets,stuff):


    #print(basedir,profile_name_sets,stuff)

    meshdir = "04_mesh"
    for profile_name in profile_name_sets:
        mesh_config = {"basedir": basedir,
                       "profile_name":profile_name,}
        write_csv_config(os.path.join(meshdir,"mesh_config.csv"), mesh_config)
        os.system(
            r"E:\AutoGrid5\autogrid132rc\bin64\iggx86_64.exe -batch -print -script D:\CodingProjects\NTRfC\ntrfc\preprocessing\mesh_creation\igg_meshcreator.py")

