from ntrfc.utils.geometry.airfoil_generators.naca_airfoil_creator import naca
from ntrfc.utils.geometry.domaingen_cascade import cascade_domain
import pyvista as pv


profilepoints2d = naca("6504", 1024, finite_te=False, half_cosine_spacing=True)

sortedPoly, per_y_upper, per_y_lower, inletPoly, outletPoly = cascade_domain(profilepoints2d,-0.9,0.2,0.4,"m",0.001,1,1, verbose=False)

p=pv.Plotter()
p.add_mesh(sortedPoly)
p.add_mesh(per_y_upper)
p.add_mesh(per_y_lower)
p.add_mesh(inletPoly)
p.add_mesh(outletPoly)
p.show()
