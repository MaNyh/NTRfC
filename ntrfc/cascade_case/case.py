import pyvista as pv
import numpy as np

from ntrfc.filehandling.mesh import load_mesh
from ntrfc.turbo.integrals import avdr
from ntrfc.geometry.plane import massflowave_plane
from ntrfc.math.vectorcalc import vecAngle


class solution:

    def __init__(self):
        self.solver = None

        self.inlet = pv.PolyData()
        self.outlet = pv.PolyData()
        self.blade = pv.PolyData()
        self.fluid = pv.UnstructuredGrid()
        self.yper_low = pv.PolyData()
        self.yper_high = pv.PolyData()
        self.zper_low = pv.PolyData()
        self.zper_high = pv.PolyData()

    def read_fluid(self, path):
        self.fluid = load_mesh(path)

    def read_inlet(self, path):
        self.inlet = load_mesh(path)

    def read_outlet(self, path):
        self.outlet = load_mesh(path)

    def read_blade(self, path):
        self.blade = load_mesh(path)

    def read_yper_low(self, path):
        self.yper_low = load_mesh(path)

    def read_yper_high(self, path):
        self.yper_high = load_mesh(path)

    def read_zper_low(self, path):
        self.zper_low = load_mesh(path)

    def read_zper_high(self, path):
        self.zper_high = load_mesh(path)

    def compute_avdr(self):
        rho_1 = massflowave_plane(self.inlet, "rho")
        mag_u_1 = massflowave_plane(self.inlet, "U")
        U_1 = np.stack([massflowave_plane(self.inlet, "u"),massflowave_plane(self.inlet, "v"),massflowave_plane(self.inlet, "w")])
        beta_1 = vecAngle(U_1, np.array([1, 0, 0]))
        rho_2 = massflowave_plane(self.outlet, "rho")
        U_2 = np.stack([massflowave_plane(self.inlet, "u"),massflowave_plane(self.inlet, "v"),massflowave_plane(self.inlet, "w")])
        mag_u_2 = massflowave_plane(self.outlet, "U")
        beta_2 = vecAngle(U_2, np.array([1, 0, 0]))
        self.avdr = avdr(rho_1, mag_u_1, beta_1, rho_2, mag_u_2, beta_2)


