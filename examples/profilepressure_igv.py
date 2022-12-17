from ntrfc.filehandling.mesh import load_mesh
from ntrfc.geometry.plane import areaave_plane
import matplotlib.pyplot as plt

solution = load_mesh("igv_005_cgns.cgns")
inlet =solution.slice(origin=(solution.bounds[0]+1e-12,0,0),normal=(1,0,0))
inlet = inlet.ptc()
pt1 = areaave_plane(inlet,"PressureStagnation")
p1 = areaave_plane(inlet,"Pressure")

outer = solution.extract_surface()

outer_slice = outer.slice(origin=solution.center,normal=(0,0,1))
regions = outer_slice.connectivity()
blade = regions.extract_cells([i for i in range(regions.number_of_cells) if regions["RegionId"][i] == 1])

camber_estimated = blade.bounds[1] - blade.bounds[0]

xs = (blade.points[::,0]-blade.bounds[0])/camber_estimated
cs = [(i-pt1)/(p1-pt1) for i in blade["Pressure"]]


plt.figure(dpi=1200)
plt.scatter(xs,cs,s=2)
plt.xlabel("$c_{ax}$")
plt.ylabel("$c_p$")
plt.xlim([0, 1])
plt.ylim([0.5, 2.5])
plt.grid()
plt.show()
