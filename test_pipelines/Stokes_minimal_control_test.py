# Imports
import os
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
from helper_functions.helper_functions import test_gradient, eval_vector_field
import imageio.v2 as imageio

# ----------------------------------------------------------------------------------------------------------------------
experiment_number = 17
np_path = f"test_pipelines/results/NS/{experiment_number}/"

# Discretization parameters
with open("parameters.json", "r") as file:
    parameters = json.load(file)
    t0 = parameters["t0"]
    T = parameters["T"]
    h = parameters["dt"]
    vis = parameters["viscosity"]
    K = parameters["buoy count"]
    mu = parameters["LR"]
    alpha = parameters["alpha"]
    mesh_boundary_x = parameters["mesh_boundary_x"]
    mesh_boundary_y = parameters["mesh_boundary_y"]

os.mkdir(np_path)
os.mkdir(np_path + "/q_data")
# ----------------------------------------------------------------------------------------------------------------------
# parameters
num_steps = 100
# ----------------------------------------------------------------------------------------------------------------------
# Mesh
gmsh.initialize()
gdim = 2
# mesh, ft, inlet_marker, wall_marker = mesh_init.create_mesh(gdim)
mesh, ft, inlet_marker, wall_marker, outlet_marker, inlet_coord, right_coord = mesh_init.create_pipe_mesh(gdim)


# ----------------------------------------------------------------------------------------------------------------------
# plotting u
x_min, x_max = 0, 2
y_min, y_max = 0, 2

# Define the number of points in each direction (resolution of the grid)
x_points = 20  # Number of points in the x direction
y_points = 20  # Number of points in the y direction

# Create linearly spaced points for x and y
x = np.linspace(x_min, x_max, x_points)
y = np.linspace(y_min, y_max, y_points)

# Create a grid from x and y
x_grid, y_grid = np.meshgrid(x, y)

# Create the array of [x, y, 0]
z = np.zeros_like(x_grid)
grid_array = np.stack([x_grid, y_grid, z], axis=-1)
grid_array = grid_array.reshape(-1, 3)
# ----------------------------------------------------------------------------------------------------------------------
# init solver instances
NS_instance = solver_classes.Navier_stokes_solver.NavierStokes(mesh, ft, inlet_marker, wall_marker, outlet_marker,
                                                               experiment_number, np_path)

# ----------------------------------------------------------------------------------------------------------------------
# init q
U_el = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
P_el = element("Lagrange", mesh.basix_cell(), 1)
W_el = mixed_element([U_el, P_el])
W = functionspace(mesh, W_el)
qp = Function(W)
q, _ = qp.split()
U, _ = W.sub(0).collapse()
dof_left = dolfinx.fem.locate_dofs_geometrical(U, lambda x: np.isclose(x[0], mesh_boundary_x[0]))
dof_right = dolfinx.fem.locate_dofs_geometrical(U, lambda x: np.isclose(x[0], mesh_boundary_x[1]))
dof_wall = dolfinx.fem.locate_dofs_geometrical(U,
                                               lambda x:
                                               np.isclose(x[1], mesh_boundary_y[0]) & np.isclose(x[1],
                                                                                                 mesh_boundary_y[1]))


def boundary_values(x):
    values = np.zeros((2, x.shape[1]))
    values[0, :] = np.where(np.isclose(x[0, :], 0.0),
                            0.1 * (1 - (x[1] - 1) ** 2),
                            0.0)
    return values


q.interpolate(boundary_values)

eval_vector_field(mesh, inlet_coord, qp.sub(0), "initial_q_vectorfield_left_boundary", np_path + "q_data",
                  False)  # plot q left side as side profile
eval_vector_field(mesh, right_coord, qp.sub(0), "initial_q_vectorfield_right_boundary", np_path + "q_data",
                  False)  # plot q left side as side profile

# ----------------------------------------------------------------------------------------------------------------------
# init plotting arrays
q_abs = []
J_array = []
q_gamma_1 = []
q_gamma_2 = []
x_array = []
grad_j_np = []
u_array = []
divs_u = []

# ----------------------------------------------------------------------------------------------------------------------
dx_ = ufl.Measure('dx', domain=mesh)
# ----------------------------------------------------------------------------------------------------------------------
# optimization loop
for i in range(num_steps):
    w_r = NS_instance.solve_stokes_step(q)
    u, p = w_r.split()
    w_adj = NS_instance.solve_adoint_stokes_step(u)
    z, _ = w_adj.split()
    q.x.array[:] = q.x.array - mu * (alpha * q.x.array - z.x.array)

    if i == 0:
        vtx_u_init = dolfinx.io.VTXWriter(mesh.comm, np_path + f"{experiment_number}_u_init.bp", w_r.sub(0).collapse(),
                                     engine="BP4")
        vtx_u_init.write(0.0)
        vtx_u_init.close()
        test_gradient(z, qp.sub(0), u, None, i, None, ft, W, alpha, inlet_marker, None, h, NS_instance,
                          None, np_path, mesh)

    div_u = form(inner(div(u), div(u)) * dx_)
    comm = u.function_space.mesh.comm
    divs_u.append(comm.allreduce(assemble_scalar(div_u), MPI.SUM))
    q_abs.append(sum(q.x.array))
# ----------------------------------------------------------------------------------------------------------------------
with open(np_path + "u_divergence.txt", "w") as text_file:
    for i, div_u_val in enumerate(divs_u):
        text_file.write("div(u) \t \t \t i  \n")
        text_file.write(f" {div_u_val} \t {i} \n")
# ----------------------------------------------------------------------------------------------------------------------
# parameter output
with open(np_path + "variables.txt", "w") as text_file:
    text_file.write("t0: %s \n" % t0)
    text_file.write("T: %s \n" % T)
    text_file.write("dt: %s \n" % h)
    text_file.write("viscosity: %s \n" % vis)
    text_file.write("buoy count: %s \n" % K)
    text_file.write("LR: %s \n" % mu)
    text_file.write("gradient descent steps: %s \n" % num_steps)
# ----------------------------------------------------------------------------------------------------------------------
# plotting
plt.plot(J_array)
plt.savefig(f"{np_path}J.png")

plt.clf()
eval_vector_field(mesh, inlet_coord, qp.sub(0), "final_q_vectorfield_left_boundary", np_path + "q_data",
                  False)  # plot q left side as side profile
eval_vector_field(mesh, right_coord, qp.sub(0), "final_q_vectorfield_right_boundary", np_path + "q_data",
                  False)  # plot q left side as side profile

# write vector field for paraview
vtx_u = dolfinx.io.VTXWriter(mesh.comm, np_path + f"{experiment_number}_u.bp", w_r.sub(0).collapse(), engine="BP4")
vtx_u.write(0.0)
vtx_u.close()
