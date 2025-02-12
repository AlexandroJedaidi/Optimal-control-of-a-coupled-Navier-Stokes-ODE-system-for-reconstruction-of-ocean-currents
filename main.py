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
from helper_functions.helper_functions import test_gradient, eval_vector_field
from basix.ufl import element, mixed_element
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# ----------------------------------------------------------------------------------------------------------------------
experiment_number = 23
np_path = f"results/experiments/{experiment_number}/"
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
    K = parameters["buoy count"]

os.mkdir(np_path)
os.mkdir(np_path + "/vector_fields")
os.mkdir(np_path + "/q_data")
# ----------------------------------------------------------------------------------------------------------------------
# parameters
num_steps = 25
# ----------------------------------------------------------------------------------------------------------------------
# Mesh
gmsh.initialize()
gdim = 2
# mesh, ft, inlet_marker, wall_marker = mesh_init.create_mesh(gdim)
mesh, ft, inlet_marker, wall_marker, outlet_marker, inlet_coord, right_coord = mesh_init.create_pipe_mesh(gdim)

if not dolfinx.has_petsc:
    print("This demo requires DOLFINx to be compiled with PETSc enabled.")
    exit(0)

# ----------------------------------------------------------------------------------------------------------------------
# helper functions


# plotting u
x_min, x_max = 0, 2
y_min, y_max = 0, 2

# Define the number of points in each direction (resolution of the grid)
x_points = 10  # Number of points in the x direction
y_points = 10  # Number of points in the y direction

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
ODE_instance = solver_classes.ODE_solver.ODE(mesh, ft, inlet_marker, wall_marker)

u_d = ODE_instance.u_d
N = ODE_instance.N
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
    # values[0, :] = np.where(np.isclose(x[0, :], 0.0), 1.0, np.where(np.isclose(x[0, :], 2.0), 1.0, 0.0))  # x-component
    # values[1, :] = np.where(np.isclose(x[0, :], 0.0), 0.0, 0.0)  # y-component
    values[0, :] = np.where(np.isclose(x[0, :], 0.0),
                            0.1 * (1 - (x[1] - 1) ** 2),  # * 3 * x[1] * (2.0 - x[1]) / (2.0 ** 2),
                            # np.where(np.isclose(x[0, :], 2.0), 0.1*(1-(x[1]-1)**2),
                            0.0  # )
                            )
    return values


qp.sub(0).interpolate(boundary_values)

eval_vector_field(mesh, inlet_coord, qp.sub(0), "initial_q_vectorfield_left_boundary",
                  np_path + "q_data")  # plot q vector field left side
eval_vector_field(mesh, right_coord, qp.sub(0), "initial_q_vectorfield_right_boundary",
                  np_path + "q_data")  # plot q vector field right side
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
# ----------------------------------------------------------------------------------------------------------------------
dx_ = ufl.Measure('dx', domain=mesh)
ds_ = ufl.Measure("ds", subdomain_data=ft)
divs_u = []
# ----------------------------------------------------------------------------------------------------------------------
# vtx_u_r = dolfinx.io.VTXWriter(mesh.comm, np_path + f"{experiment_number}_u_reference.bp", w_r.sub(0).collapse(), engine="BP4")
# vtx_u_r.write(0.0)
# vtx_u_r.close()
# ----------------------------------------------------------------------------------------------------------------------
# optimization loop
for i in range(num_steps):
    w_r = NS_instance.solve_stokes_step(qp.sub(0))
    u_r, p_r = w_r.split()

    w = NS_instance.state_solving_step(qp.sub(0), u_r, i)  # step one

    u, p = w.split()
    eval_vector_field(mesh, grid_array, u, f"u_vectorfield_{str(i)}_AFTER_ITERATION",
                      np_path + "vector_fields")  # plot vector fields

    x = ODE_instance.ode_solving_step(u)  # step two
    lam_2 = ODE_instance.adjoint_ode_solving_step(u)  # step three
    w_adj, J, u_values = NS_instance.adjoint_state_solving_step(u, lam_2, x, h, u_d, qp.sub(0), None)  # step four

    u_adj, p_adj = w_adj.split()

    if i == num_steps - 1 or i == 0:
        div_u = form(div(u) ** 2 * dx_)
        comm = u.function_space.mesh.comm
        divs = comm.allreduce(assemble_scalar(div_u), MPI.SUM)
        divs_u.append(divs)

    #grad_j_FD = test_gradient(u_adj, qp.sub(0), u, x, i, u_r, ft, W, alpha, inlet_marker, u_d, h, NS_instance,
                  #ODE_instance, np_path, mesh)

    # grad_j = qp.sub(0).x.array[:] - u_adj.x.array[:]
    grad_j_func = alpha * qp.sub(0) - u_adj
    red_grad = form(dot(grad_j_func, qp.sub(0)) * ds_(inlet_marker))
    J_grad_red = assemble_scalar(red_grad)
    grad_j_np.append(J_grad_red)

    qp.sub(0).x.array[:] = qp.sub(0).x.array[:] - mu * (alpha * qp.sub(0).x.array[:] - u_adj.x.array[:])
    # qp.sub(0).x.array[:] = qp.sub(0).x.array - mu * grad_j_FD.x.array

    q_abs.append(sum(q.x.array))
    J_array.append(J)
    x_array.append(x)
    u_array.append(u_values)
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
print("plotting")
plt.plot(J_array)
plt.savefig(f"{np_path}J.png")

plt.clf()

plt.plot(grad_j_np)
plt.savefig(f"{np_path}grad_J.png")

plt.clf()

os.mkdir(np_path + "buoy_movements")
os.mkdir(np_path + "buoy_movements/frames")
for k, x_ in enumerate(x_array):
    color = plt.cm.rainbow(np.linspace(0, 1, K))
    for i, x_buoy in enumerate(x_):
        x_coord = x_buoy[:, 0]
        y_coord = x_buoy[:, 1]
        plt.xlim(0.0, 2)
        plt.ylim(0.0, 2)
        plt.plot([x_buoy[0, 0], x_buoy[0, 0] + 1 / (np.pi)], [x_buoy[0, 1], x_buoy[0, 1]], label="x0 and xT for u_D",
                 color=color[i], linewidth=8)
        plt.plot(x_coord, y_coord, label=f"buoy_{i}_movement", color="b")

    plt.savefig(f"{np_path}buoy_movements/frames/buoy_movement_{k}.png")
    plt.clf()

# create gif
print("creating videos")
video_path = f"{np_path}buoy_movements/buoy_animation.mp4"
with imageio.get_writer(video_path, fps=10, format='mp4') as writer:
    for i in range(num_steps):
        frames_path = os.path.join(f"{np_path}buoy_movements/frames", f"buoy_movement_{i}.png")
        image = imageio.imread(frames_path)
        writer.append_data(image)

ud = u_d.reshape(u_d.shape[0] * u_d.shape[1], u_d.shape[2])
u_values = np.array(u_array)
err = []
for u_vals in u_values:
    error = np.linalg.norm(u_vals - u_d, axis=2) ** 2
    error = np.sum(np.sum(h * error, axis=1))
    err.append(error)
plt.plot(err, color="blue")
plt.savefig(f"{np_path}buoy_0_velocity_error.png")
plt.clf()

# print inlet velocity field
eval_vector_field(mesh, inlet_coord, qp.sub(0), "final_q_vectorfield_left_boundary", np_path + "q_data")
eval_vector_field(mesh, right_coord, qp.sub(0), "final_q_vectorfield_right_boundary", np_path + "q_data")
eval_vector_field(mesh, inlet_coord, qp.sub(0), "final_q_vectorfield_left_boundary", np_path + "q_data",
                  False)  # plot q left side as side profile
eval_vector_field(mesh, right_coord, qp.sub(0), "final_q_vectorfield_right_boundary", np_path + "q_data",
                  False)  # plot q left side as side profile

eval_vector_field(mesh, grid_array, u, "final_u_vectorfield", np_path + "vector_fields")

print("write u")
vtx_u = dolfinx.io.VTXWriter(mesh.comm, np_path + f"{experiment_number}_u.bp", w.sub(0).collapse(), engine="BP4")
vtx_u.write(0.0)
vtx_u.close()

print("write q")
vtx_q = dolfinx.io.VTXWriter(mesh.comm, np_path + f"{experiment_number}_q.bp", qp.sub(0).collapse(), engine="BP4")
vtx_q.write(0.0)
vtx_q.close()

print("write p")
vtx_p = dolfinx.io.VTXWriter(mesh.comm, np_path + f"{experiment_number}_p.bp", w.sub(1).collapse(), engine="BP4")
vtx_p.write(0.0)
vtx_p.close()
