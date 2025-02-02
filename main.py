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

# ----------------------------------------------------------------------------------------------------------------------
experiment_number = 117
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

os.mkdir(np_path)
os.mkdir(np_path + "/vector_fields")
os.mkdir(np_path + "/q_data")
# ----------------------------------------------------------------------------------------------------------------------
# parameters
num_steps = 50
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
def eval_vector_field(grid, coordinates, func, func_name, path, vector_field=True):
    if vector_field:
        bb_tree = dolfinx.geometry.bb_tree(grid, grid.topology.dim)
        func_eval = []
        for point in coordinates:
            cells = []
            points_on_proc = []
            cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, point)
            colliding_cells = dolfinx.geometry.compute_colliding_cells(grid, cell_candidates, point)
            if len(colliding_cells.links(0)) > 0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(0)[0])
            points_on_proc = np.array(points_on_proc, dtype=np.float64)
            if len(colliding_cells) < 0:
                print("no colliding cells")
                from IPython import embed
                embed()
            func_eval.append(func.eval(points_on_proc, cells))
        y_axis = []
        for i, coord in enumerate(func_eval):
            x_ = coordinates[i][0]
            y_ = coordinates[i][1]
            y_axis.append(y_)
            u_vec = func_eval[i][0]
            v_vec = func_eval[i][1]
            plt.quiver(x_, y_, u_vec, v_vec, 1, angles="xy", scale_units="xy", scale=0.6, cmap="viridis")
        plt.savefig(f"{path}/{func_name}_vectorfield.png")
        plt.clf()
    else:
        coordinates = np.sort(coordinates, axis=0)
        bb_tree = dolfinx.geometry.bb_tree(grid, grid.topology.dim)
        func_eval = []
        for point in coordinates:
            cells = []
            points_on_proc = []
            cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, point)
            colliding_cells = dolfinx.geometry.compute_colliding_cells(grid, cell_candidates, point)
            if len(colliding_cells.links(0)) > 0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(0)[0])
            points_on_proc = np.array(points_on_proc, dtype=np.float64)
            if len(colliding_cells) < 0:
                print("no colliding cells")
                from IPython import embed
                embed()
            func_eval.append(func.eval(points_on_proc, cells))
        y_axis = []
        for i, coord in enumerate(func_eval):
            y_ = coordinates[i][1]
            y_axis.append(y_)
        plt.plot(y_axis, [item[0] for item in func_eval])
        plt.savefig(f"{path}/{func_name}_flow_profile.png")
        plt.clf()


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
U_el = element("Lagrange", mesh.basix_cell(), 5, shape=(mesh.geometry.dim,))
P_el = element("Lagrange", mesh.basix_cell(), 4)
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
                            0.1 ,#* 3 * x[1] * (2.0 - x[1]) / (2.0 ** 2),
                            0.0)
    return values


q.interpolate(boundary_values)

eval_vector_field(mesh, inlet_coord, q, "initial_q_vectorfield_left_boundary",
                  np_path + "q_data")  # plot q vector field left side
eval_vector_field(mesh, right_coord, q, "initial_q_vectorfield_right_boundary",
                  np_path + "q_data")  # plot q vector field right side
eval_vector_field(mesh, inlet_coord, q, "initial_q_vectorfield_left_boundary", np_path + "q_data",
                  False)  # plot q left side as side profile
eval_vector_field(mesh, right_coord, q, "initial_q_vectorfield_right_boundary", np_path + "q_data",
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
def evalutate_fuct(fct, points):
    bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    zeros = np.zeros((*points.shape[:2], 1))
    new_points = np.concatenate((points, zeros), axis=-1)

    point_shape = new_points.shape
    new_points = new_points.reshape(point_shape[0] * point_shape[1], point_shape[2])

    cells = []
    points_on_proc = []
    cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, new_points)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates,
                                                               new_points)
    for i, point in enumerate(new_points):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])

    return fct.eval(points_on_proc, cells)


def test_gradient(lam_, q_func, u_red, x_red, opt_iter,u_r):
    ds_ = ufl.Measure("ds", subdomain_data=ft)
    dq, _ = Function(W).split()
    dq.x.array[:] = 1
    h_ = np.array([10 ** -(i + 1) for i in range(14)])

    grad_j_funcp = Function(W)
    grad_j_func, _ = grad_j_funcp.split()
    grad_j_func.x.array[:] = alpha * q_func.x.array[:] - lam_.x.array[:]
    red_grad = form(dot(grad_j_func, dq) * ds_(inlet_marker))
    comm_red = dq.function_space.mesh.comm
    J_grad_red = comm_red.allreduce(assemble_scalar(red_grad), MPI.SUM)

    ud = u_d.reshape(u_d.shape[0] * u_d.shape[1], u_d.shape[2])
    u_values_red = evalutate_fuct(u_red, x_red)
    qnorm = form(dot(q_func, q_func) * ds_(inlet_marker))
    comm = q_func.function_space.mesh.comm
    E = comm.allreduce(assemble_scalar(qnorm), MPI.SUM)
    J_red = 0.5 * h*sum((np.linalg.norm(u_values_red - ud, axis=1) ** 2)) + (alpha / 2) * E

    dJ = []
    for h_i in h_:
        qhat,_ = Function(W).split()
        qhat.x.array[:] = q_func.x.array[:] + h_i * dq.x.array[:]
        w_, W_ = NS_instance.state_solving_step(qhat,u_r)
        u_, p_ = w_.split()
        x_ = ODE_instance.ode_solving_step(u_)

        qnorm = form(dot(qhat, qhat) * ds_(inlet_marker))
        comm = qhat.function_space.mesh.comm
        E = comm.allreduce(assemble_scalar(qnorm), MPI.SUM)

        u_values_ = evalutate_fuct(u_, x_)

        dJ.append((0.5 * h * sum((np.linalg.norm(u_values_ - ud, axis=1) ** 2)) + (alpha / 2) * E - J_red) / h_i)

    with open(np_path + f"grad_J_error_{opt_iter}.txt", "w") as text_file:
        text_file.write("reduced Gradient j \t \t approximated gradient J \t Error \t \t \t h_i \n")
        for i, h_i in enumerate(h_[:-1]):
            text_file.write(f" {J_grad_red} \t {dJ[i]} \t {dJ[i] - J_grad_red} \t {h_i} \n")


# ----------------------------------------------------------------------------------------------------------------------
dx_ = ufl.Measure('dx', domain=mesh)
divs_u = []
# ----------------------------------------------------------------------------------------------------------------------
# vtx_u_r = dolfinx.io.VTXWriter(mesh.comm, np_path + f"{experiment_number}_u_reference.bp", w_r.sub(0).collapse(), engine="BP4")
# vtx_u_r.write(0.0)
# vtx_u_r.close()
# ----------------------------------------------------------------------------------------------------------------------
# optimization loop
for i in range(num_steps):
    q_gamma_1.append(sum(q.x.array[dof_left]) + sum(q.x.array[dof_right]))
    q_gamma_2.append(sum(q.x.array[dof_wall]))

    w_r = NS_instance.solve_stokes_step(q)
    u_r, p_r = w_r.split()

    w, W = NS_instance.state_solving_step(q, u_r)  # step one

    u, p = w.split()
    #eval_vector_field(mesh, grid_array, u, f"u_vectorfield_{str(i)}_AFTER_ITERATION",
    #                  np_path + "vector_fields")  # plot vector fields

    x = ODE_instance.ode_solving_step(u)  # step two
    lam_2 = ODE_instance.adjoint_ode_solving_step(u)  # step three
    w_adj, J, u_values = NS_instance.adjoint_state_solving_step(u, lam_2, x, h, u_d, q, u_r)  # step four

    u_adj, p_adj = w_adj.split()

    # if i == num_steps-1 or i == 0:
    #     div_u = form(dot(div(u), div(u)) * dx_)
    #     comm = u.function_space.mesh.comm
    #     divs_u.append(comm.allreduce(assemble_scalar(div_u), MPI.SUM))
    #     test_gradient(u_adj, q, u, x, i,u_r)
    #
    # grad_j = q.x.array[:] - u_adj.x.array[:]
    # grad_j_np.append(np.linalg.norm(grad_j))

    q.x.array[:] = q.x.array[:] - mu * (alpha * q.x.array[:] - u_adj.x.array[:])

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
plt.plot(J_array)
plt.savefig(f"{np_path}J.png")

plt.clf()

# plt.plot(q_gamma_1)
# plt.savefig(f"{np_path}q_Gamma1.png")
#
# plt.clf()
#
# plt.plot(q_gamma_2)
# plt.savefig(f"{np_path}q_Gamma2.png")
#
# plt.clf()

# plt.plot(grad_j_np)
# plt.savefig(f"{np_path}grad_J.png")
#
# plt.clf()

os.mkdir(np_path + "/buoy_movements")
for k, x_ in enumerate(x_array):
    for i, x_buoy in enumerate(x_):
        x_coord = x_buoy[:, 0]
        y_coord = x_buoy[:, 1]
        plt.xlim(0,2)
        plt.ylim(0,2)
        plt.plot(x_coord, y_coord, label=f"buoy_{i}_movement", **{'color': 'lightsteelblue', 'marker': 'o'})
        plt.savefig(f"{np_path}buoy_movements/buoy_{i}_{k}_movement.png")
        plt.clf()

ud = u_d.reshape(u_d.shape[0] * u_d.shape[1], u_d.shape[2])
extended_ud = np.tile(ud[np.newaxis, :, :], (num_steps, 1, 1))
u_values = np.array(u_array)

error = np.linalg.norm(u_values - extended_ud, axis=2)
error = np.sum(h * error, axis=1)
plt.plot(error, color="blue")
plt.savefig(f"{np_path}buoy_0_velocity_error.png")
plt.clf()

# print inlet velocity field
eval_vector_field(mesh, inlet_coord, q, "final_q_vectorfield_left_boundary", np_path + "q_data")
eval_vector_field(mesh, right_coord, q, "final_q_vectorfield_right_boundary", np_path + "q_data")
eval_vector_field(mesh, inlet_coord, q, "final_q_vectorfield_left_boundary", np_path + "q_data",
                  False)  # plot q left side as side profile
eval_vector_field(mesh, right_coord, q, "final_q_vectorfield_right_boundary", np_path + "q_data",
                  False)  # plot q left side as side profile

eval_vector_field(mesh, grid_array, u, "final_u_vectorfield", np_path + "vector_fields")

# write vector field for paraview
# with dolfinx.io.XDMFFile(mesh.comm, f"{np_path}/paraview/u_final.xdmf", "w") as file:
#     file.write_mesh(mesh)
#     file.write_function(u)
vtx_u = dolfinx.io.VTXWriter(mesh.comm, np_path + f"{experiment_number}_u.bp", w.sub(0).collapse(), engine="BP4")
vtx_u.write(0.0)
vtx_u.close()
