import numpy as np
import gstools as gs
import pyvista as pv
from tqdm import tqdm


# fix the seed for reproducibility
seed = 20170521

x = y = z = np.linspace(-0.05, 0.05, 140)
tres = 256
tlen = 4
t = np.linspace(0, tlen, tlen*tres)

# an exponential variogram with a corr. lengths of 5km, 5km, and 2d
model = gs.Gaussian(
    temporal=False, spatial_dim=3, var=1, len_scale=[0.01,0.01,0.1],anis=0.5, angles=[0, 0, 0]
)
# create a spatial random field instance
srf = gs.SRF(model, seed=seed, generator="VectorField",)
print("initializing")
srf.structured((x, y,  0))
mesh = srf.to_pyvista()
mesh.clear_data()

print("computing")
# the Gaussian random field
field = srf.structured((x, y,  t))

print("plotting")
p = pv.Plotter(off_screen=True,window_size=[2400, 2400])
p.open_gif("wave.gif")
p.add_mesh(mesh)
p.view_isometric()
for frame in tqdm(range(len(t))):
    p.clear()
    fieldvals = srf.field[:, :,:, frame]
    mesh["Velocity"] = fieldvals.reshape(3, -1).T
    p.add_mesh(mesh.glyph(orient="Velocity",factor=0.001))
    p.add_text(f"Frame: {frame} / {len(t)}")
    p.write_frame()

p.close()
print("done")
