import pyvista as pv

from ntrfc.cascade_case.solution.case_modules.postprocessing import postprocessing
from ntrfc.cascade_case.solution.case_modules.proberegistry import proberegistry
from ntrfc.cascade_case.solution.case_modules.sliceregistry import sliceregistry
from ntrfc.filehandling.mesh import load_mesh


class GenericCascadeCase(postprocessing, proberegistry, sliceregistry):
    def __init__(self):
        super().__init__()
        self.solver = None

        self.inlet = pv.PolyData()
        self.outlet = pv.PolyData()
        self.blade = pv.PolyData()
        self.fluid = pv.UnstructuredGrid()
        self.yper_low = pv.PolyData()
        self.yper_high = pv.PolyData()
        self.zper_low = pv.PolyData()
        self.zper_high = pv.PolyData()

        self.probes = proberegistry()
        self.sliceseries = sliceregistry()

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
