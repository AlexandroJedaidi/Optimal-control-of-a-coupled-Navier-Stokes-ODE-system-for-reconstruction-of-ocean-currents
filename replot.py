from fenics import *
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import mshr
import os
import time
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300

plt.rcParams["font.family"] = "TeX Gyre Heros"
plt.rcParams["mathtext.fontset"] = "cm"

experiment = 255
ud_experiment = 30  # "6_4b"
np_path = f"results/dolfin/OCP/replots/{experiment}/"
os.mkdir(np_path)
# ----------------------------------------------------------------------------------------------------------------------
# plot mesh
Nx = 32
left_x = 0.0
left_y = 2.0
right_x = 2.0
right_y = 2.0
mesh = RectangleMesh(Point(left_x, left_x), Point(right_x, right_y), Nx, Nx)
plt.title(r"discretized domain $\Omega_h$")
plt.xlabel("$x$")
plt.ylabel("$y$")
plot(mesh)
plt.savefig(f"{np_path}mesh.png")
plt.clf()

mesh_boundary = [[[0.0, 2.0], [0.0, 0.0]],
                 [[0.0, 0.0], [0.0, 2.0]],
                 [[0.0, 2.0], [2.0, 2.0]],
                 [[2.0, 2.0], [2.0, 0.0]]]
# ----------------------------------------------------------------------------------------------------------------------
ud_loaded = Function(W.sub(0).collapse())
with XDMFFile(f"results/dolfin/OCP/ud_construction/{ud_experiment}/paraview/velocity.xdmf") as infile:
    infile.read_checkpoint(ud_loaded, "u")