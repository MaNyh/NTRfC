import numpy as np
from ntrfc.geometry.plane import massflowave_plane
from ntrfc.math.vectorcalc import vecAngle
from ntrfc.turbo.integrals import avdr


class postprocessing:
    def compute_avdr(self):
        rho_1 = massflowave_plane(self.inlet, "rho")
        mag_u_1 = massflowave_plane(self.inlet, "U")
        U_1 = np.stack([massflowave_plane(self.inlet, "u"), massflowave_plane(self.inlet, "v"),
                        massflowave_plane(self.inlet, "w")])
        beta_1 = vecAngle(U_1, np.array([1, 0, 0]))
        rho_2 = massflowave_plane(self.outlet, "rho")
        U_2 = np.stack([massflowave_plane(self.inlet, "u"), massflowave_plane(self.inlet, "v"),
                        massflowave_plane(self.inlet, "w")])
        mag_u_2 = massflowave_plane(self.outlet, "U")
        beta_2 = vecAngle(U_2, np.array([1, 0, 0]))
        self.avdr = avdr(rho_1, mag_u_1, beta_1, rho_2, mag_u_2, beta_2)
