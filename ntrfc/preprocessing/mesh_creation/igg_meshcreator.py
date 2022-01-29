import os
import csv

meshdir = "04_mesh"

if not os.path.isdir(meshdir):
    os.mkdir(meshdir)

os.chdir(meshdir)


def open_csvconfig(fname="mesh_config.csv"):
    with open(fname, 'r') as f:
        mesh_configreader = csv.reader(f, delimiter=' ')
        mesh_config = {}
        for line in mesh_configreader:
            mesh_config[line[0]] = line[1]
    return mesh_config


def openTecplotFile(path):
    values = []
    var = []
    zones = []

    zone_bool = -1
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('VARIABLES'):
                line_string = line.replace('\n', '').split('=')[-1].split(',')
                for string in line_string:
                    var.append(string.replace('"', ''))

            if line.startswith('ZONE'):

                zone_bool = zone_bool + 1

                zones.append(line.replace('\n', '').split('=')[-1].replace('"', ''))
                list_01 = []
                for i in range(len(var)):
                    list_01.append([])

                values.append(list_01)

            if line.startswith('ZONE') or line.startswith('VARIABLES') or line.startswith('TITLE'):
                pass

            else:
                line_values = line.split()
                for i in range(len(line_values)):
                    values[zone_bool][i].append(float(line_values[i]))

    return values


#########################################################################
#                                                                       #
#                           profile points                              #
#                                                                       #
#########################################################################
mesh_config = open_csvconfig(fname="mesh_config.csv")

blocks = openTecplotFile(os.path.join(mesh_config["basedir"],"03_meshgeometry", mesh_config["profile_name"] + "_blocks.geom"))
domain = openTecplotFile(os.path.join(mesh_config["basedir"],"03_meshgeometry", mesh_config["profile_name"] + "_domain.geom"))

x_ps = domain[4][0]
y_ps = domain[4][1]
x_ss = domain[5][0]
y_ss = domain[5][1]

#########################################################################
#                                                                       #
#                           geometry data                               #
#                                                                       #
#########################################################################

y_lower_x = domain[3][0]
y_lower_y = domain[3][1]
y_upper_x = domain[2][0]
y_upper_y = domain[2][1]
inlet_x = domain[0][0]
inlet_y = domain[0][1]
outlet_x = domain[1][0]
outlet_y = domain[1][1]

#########################################################################
#                                                                       #
#                           geometry curves                             #
#                                                                       #
#########################################################################

le_ogrid_x = blocks[0][0]
le_ogrid_y = blocks[0][1]
te_ogrid_x = blocks[1][0]
te_ogrid_y = blocks[1][1]
ogrid_inlet_x = blocks[2][0]
ogrid_inlet_y = blocks[2][1]
ogrid_oulet_x = blocks[3][0]
ogrid_oulet_y = blocks[3][1]
ogridline_x = blocks[4][0]
ogridline_y = blocks[4][1]
ps_x0_ogrid_line_x = blocks[5][0]
ps_x0_ogrid_line_y = blocks[5][1]
ps_x1_ogrid_line_x = blocks[6][0]
ps_x1_ogrid_line_y = blocks[6][1]
ss_x0_ogrid_line_x = blocks[7][0]
ss_x0_ogrid_line_y = blocks[7][1]
ss_x1_ogrid_line_x = blocks[8][0]
ss_x1_ogrid_line_y = blocks[8][1]
ylower_ogrid_x0_x = blocks[9][0]
ylower_ogrid_x0_y = blocks[9][1]
yupper_ogrid_x0_x = blocks[10][0]
yupper_ogrid_x0_y = blocks[10][1]
ylower_ogrid_x1_x = blocks[11][0]
ylower_ogrid_x1_y = blocks[11][1]
yupper_ogrid_x1_x = blocks[12][0]
yupper_ogrid_x1_y = blocks[12][1]

#########################################################################
#                                                                       #
#                           geometry points                             #
#                                                                       #
#########################################################################

#########################################################################
#                                                                       #
#                           geometry blocks                             #
#                                                                       #
#########################################################################

#########################################################################
#                                                                       #
#                           point distrubution                          #
#                                                                       #
#########################################################################


#########################################################################
#                                                                       #
#                           smoothing                                   #
#                                                                       #
#########################################################################


save_project("mesh.igg")
export_FLUENT("fluent.msh")

os.chdir("..")
