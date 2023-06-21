import numpy as np

from ntrfc.geometry.plane import massflowave_plane
from ntrfc.math.vectorcalc import vecAngle, vecAbs
from ntrfc.turbo.bladeloading import calc_inflow_cp
from ntrfc.turbo.integrals import avdr


class PostProcessing:
    def compute_avdr_inout_massave(self):

        self.mesh_dict["inlet"]["u"] = self.mesh_dict["inlet"]["U"][::, 0]
        self.mesh_dict["inlet"]["v"] = self.mesh_dict["inlet"]["U"][::, 1]
        self.mesh_dict["inlet"]["w"] = self.mesh_dict["inlet"]["U"][::, 2]

        self.mesh_dict["outlet"]["u"] = self.mesh_dict["outlet"]["U"][::, 0]
        self.mesh_dict["outlet"]["v"] = self.mesh_dict["outlet"]["U"][::, 1]
        self.mesh_dict["outlet"]["w"] = self.mesh_dict["outlet"]["U"][::, 2]
        rho_1 = massflowave_plane(self.mesh_dict["inlet"], "rho")
        mag_u_1 = vecAbs(
            np.array([massflowave_plane(self.mesh_dict["inlet"], "u"), massflowave_plane(self.mesh_dict["inlet"], "v"),
                      massflowave_plane(self.mesh_dict["inlet"], "w")]))
        U_1 = np.stack(
            [massflowave_plane(self.mesh_dict["inlet"], "u"), massflowave_plane(self.mesh_dict["inlet"], "v"),
             massflowave_plane(self.mesh_dict["inlet"], "w")])
        beta_1 = vecAngle(U_1, np.array([1, 0, 0]))
        rho_2 = massflowave_plane(self.mesh_dict["outlet"], "rho")
        U_2 = np.stack(
            [massflowave_plane(self.mesh_dict["outlet"], "u"), massflowave_plane(self.mesh_dict["outlet"], "v"),
             massflowave_plane(self.mesh_dict["outlet"], "w")])
        mag_u_2 = vecAbs(np.array(
            [massflowave_plane(self.mesh_dict["outlet"], "u"), massflowave_plane(self.mesh_dict["outlet"], "v"),
             massflowave_plane(self.mesh_dict["outlet"], "w")]))
        beta_2 = vecAngle(U_2, np.array([1, 0, 0]))
        self.avdr = avdr(rho_1, mag_u_1, beta_1, rho_2, mag_u_2, beta_2)

    def blade_loading(self):
        sspoints = self.domainparams.sspoly.points
        pspoints = self.domainparams.pspoly.points

        inlet = self.mesh_dict["inlet"]
        inlet["u"] = inlet["U"][::, 0]
        inlet["v"] = inlet["U"][::, 1]
        inlet["w"] = inlet["U"][::, 2]
        p1 = massflowave_plane(inlet, valname="p", rhoname="rho", velocityname="U")
        rho = massflowave_plane(inlet, valname="rho", rhoname="rho", velocityname="U")
        u = massflowave_plane(inlet, valname="u", rhoname="rho", velocityname="U")
        v = massflowave_plane(inlet, valname="v", rhoname="rho", velocityname="U")
        w = massflowave_plane(inlet, valname="w", rhoname="rho", velocityname="U")
        U = vecAbs([u, v, w])
        pt1 = p1 + 1 / 2 * rho * U ** 2

        bladepoly = self.domainparams.profile_points

        ssmeshpointids = [bladepoly.find_closest_point(pt) for pt in sspoints]
        psmeshpointids = [bladepoly.find_closest_point(pt) for pt in pspoints]

        ssmeshpoints = bladepoly.extract_points(ssmeshpointids)
        psmeshpoints = bladepoly.extract_points(psmeshpointids)

        ind_le = self.domainparams.leading_edge_index
        ind_te = self.domainparams.trailing_edge_index
        ssmeshpoints.points -= bladepoly.points[ind_le]
        psmeshpoints.points -= bladepoly.points[ind_le]
        bladepoly.points -= bladepoly.points[ind_le]
        camber_length = vecAbs(bladepoly.points[ind_le] - bladepoly.points[ind_te])

        camber_angle = self.domainparams.stagger_angle

        ssmeshpoints.rotate_z(camber_angle, inplace=True)
        psmeshpoints.rotate_z(camber_angle, inplace=True)

        ps_xc = np.zeros(psmeshpoints.number_of_points)
        ps_cp = np.zeros(psmeshpoints.number_of_points)

        for idx, pts1 in enumerate(psmeshpoints.points):
            ps_xc[idx] = pts1[0] / camber_length
            ps_cp[idx] = calc_inflow_cp(psmeshpoints.point_data["p"][idx], pt1, p1)

        ss_xc = np.zeros(ssmeshpoints.number_of_points)
        ss_cp = np.zeros(ssmeshpoints.number_of_points)

        for idx, pts1 in enumerate(ssmeshpoints.points):
            ss_xc[idx] = pts1[0] / camber_length
            ss_cp[idx] = calc_inflow_cp(ssmeshpoints.point_data["p"][idx], pt1, p1)

        # todo: add plotting to file
        # plt.figure()
        # plt.title("blade loading")
        # plt.scatter(ss_xc, ss_cp, label="suction side")
        # plt.scatter(ps_xc, ps_cp, label="pressure side")
        # plt.xlabel("$x/c_{ax}$")
        # plt.ylabel("$c_{p}$")
        # plt.grid()
        # plt.legend()
        # plt.show()

        return 0
