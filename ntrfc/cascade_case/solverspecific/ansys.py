import numpy as np
import pandas as pd
import pyvista as pv

from ntrfc.cascade_case.case import solution
from ntrfc.filehandling.mesh import load_mesh


class ansys_solution(solution):
    def __init__(self):
        super().__init__()
        self.solver = "ansys"

    def read_fluid(self, path):
        self.fluid = load_mesh(path)

    def read_inlet(self, path):
        self.inlet = read_ansys_csv(path)

    def read_outlet(self, path):
        self.outlet = read_ansys_csv(path)

    def read_blade(self, path):
        self.blade = load_mesh(path)

    def read_yper_low(self, path):
        self.yper_low = read_ansys_csv(path)

    def read_yper_high(self, path):
        self.yper_high = read_ansys_csv(path)

    def read_zper_low(self, path):
        self.zper_low = read_ansys_csv(path)

    def read_zper_high(self, path):
        self.zper_high = read_ansys_csv(path)


def read_ansys_csv(path):
    dats = {
        "p": " Pressure [ Pa ]",
        "T": " Temperature [ K ]",
        "U": " Velocity [ m s^-1 ]",
        "u": " Velocity u [ m s^-1 ]",
        "v": " Velocity v [ m s^-1 ]",
        "w": " Velocity w [ m s^-1 ]",
        "rho": " Density [ kg m^-3 ]",
    }

    data = pd.read_csv(path, skiprows=5)
    xs, ys, zs = data["X [ m ]"], data[" Y [ m ]"], data[" Z [ m ]"]
    points = np.stack([xs, ys, zs]).T
    poly = pv.PolyData(points)

    for k, v in dats.items():
        poly[k] = data[v]

    surface = poly.delaunay_2d()
    surface = surface.point_data_to_cell_data()
    return surface



