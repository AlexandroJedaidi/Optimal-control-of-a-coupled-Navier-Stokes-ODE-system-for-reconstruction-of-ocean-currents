# Imports
import gmsh
import dolfinx
from petsc4py import PETSc
import ufl
from mpi4py import MPI
import json
from dolfinx.fem import (Constant, Function, functionspace,
                         assemble_scalar, dirichletbc, form, locate_dofs_topological, set_bc)
from ufl import (FacetNormal, Identity, Measure, TestFunction, TrialFunction,
                 as_vector, div, dot, inner, lhs, grad, nabla_grad, rhs, sym, system, SpatialCoordinate, inv,
                 sqrt, transpose, tr)
import numpy as np
import numpy.typing as npt
import mesh_init
import solver_classes.Navier_stokes_solver
import solver_classes.ODE_solver
from basix.ufl import element, mixed_element
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------
experiment_number = 2
# Discretization parameters
with open("parameters.json", "r") as file:
    parameters = json.load(file)
    t0 = parameters["t0"]
    T = parameters["T"]
    dt = parameters["dt"]
    vis = parameters["viscosity"]
    K = parameters["buoy count"]
    mu = parameters["LR"]

    # num_steps = int(T / dt)
# ----------------------------------------------------------------------------------------------------------------------
# Mesh
gmsh.initialize()
gdim = 2
# mesh, ft, inlet_marker, wall_marker = mesh_init.create_mesh(gdim)
mesh, ft, inlet_marker, wall_marker, outlet_marker = mesh_init.create_pipe_mesh(gdim)
# ----------------------------------------------------------------------------------------------------------------------
NS_instance = solver_classes.Navier_stokes_solver.NavierStokes(mesh, ft, inlet_marker, wall_marker, outlet_marker,
                                                               experiment_number)
ODE_instance = solver_classes.ODE_solver.ODE(mesh, ft, inlet_marker, wall_marker)
U_el = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
P_el = element("Lagrange", mesh.basix_cell(), 1)
W_el = mixed_element([U_el, P_el])
W = functionspace(mesh, W_el)
num_steps = 100
qp = Function(W)
q, _ = qp.split()
# q.x.array[:] = 0.1
q_abs = []
J_array = []
q_gamma_1 = []
x_array = []
U, _ = W.sub(0).collapse()
dofs_q = locate_dofs_topological(U, mesh.topology.dim - 1, ft.find(inlet_marker))
for i in range(num_steps):
    q_gamma_1.append(sum(q.x.array[dofs_q]))
    w, W = NS_instance.state_solving_step(q)
    u, p = w.split()
    x, h, N, u_d = ODE_instance.ode_solving_step(u)
    lam_2 = ODE_instance.adjoint_ode_solving_step(u)
    w_adj, J = NS_instance.adjoint_state_solving_step(u, lam_2, x, h, N, u_d, q)
    u_adj, p_adj = w_adj.split()

    grad_j = q.x.array - u_adj.x.array
    q.x.array[:] = (1 - mu) * q.x.array - mu * u_adj.x.array

    q_abs.append(sum(q.x.array))
    J_array.append(J)
    x_array.append(x[0])

plt.plot(q_abs)
plt.savefig("q_abs_100_it.png")

plt.clf()

plt.plot(J_array)
plt.savefig("J_100_it.png")

plt.clf()

plt.plot(q_gamma_1)
plt.savefig("q_Gamma1_100_it.png")

plt.clf()

# from IPython import embed; embed()
x = np.array(x_array).reshape(100 * num_steps, 2)
plt.plot(x[:, 0], x[:, 1])
plt.plot(u_d[0][:, 0], u_d[0][:, 1], marker=".")
plt.savefig("0_buoy.png")

# x = np.array(x_array).reshape(10,100,100,2)
# for id, buoy_coord in enumerate(x):
#    plt.plot(buoy_coord[:, 0], buoy_coord[:, 1], label="buoy: " + str(id), color="blue")
#    plt.plot(u_d[id][:, 0], u_d[id][:, 1], label="wanted position: " + str(id), color="red")
#    plt.legend(loc="lower right")
#    plt.savefig(str(id) + "_buoy.png")
#    plt.clf()

from IPython import embed

embed()

with dolfinx.io.VTXWriter(MPI.COMM_WORLD, "velocity.bp", [u], engine="BP4") as vtx:
    vtx.write(0.0)
