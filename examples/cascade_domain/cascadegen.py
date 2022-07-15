from ntrfc.utils.geometry.airfoil_generators.naca_airfoil_creator import naca
from ntrfc.utils.geometry.domaingen_cascade import cascade_2d_domain,cascade_3d_domain,gmsh_3d_domain
import pyvista as pv


profilepoints2d = naca("6504", 1024, finite_te=False, half_cosine_spacing=True)

sortedPoly, per_y_upper, per_y_lower, inletPoly, outletPoly = cascade_2d_domain(profilepoints2d,-0.9,0.2,0.4,"m",0.001,1, verbose=False)

# p=pv.Plotter()
# p.add_mesh(sortedPoly)
# p.add_mesh(per_y_upper)
# p.add_mesh(per_y_lower)
# p.add_mesh(inletPoly)
# p.add_mesh(outletPoly)
# p.show()

sortedPoly_3d,per_y_upper_3d,per_y_lower_3d,inletPoly_3d,outletPoly_3d ,per_z_lower,per_z_upper= cascade_3d_domain(sortedPoly, per_y_upper, per_y_lower, inletPoly, outletPoly, 0.5, 2, verbose=False)

gmsh_3d_domain(sortedPoly_3d,per_y_upper_3d,per_y_lower_3d,inletPoly_3d,outletPoly_3d,per_z_lower,per_z_upper )
