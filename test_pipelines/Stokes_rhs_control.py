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
                 sqrt, transpose, tr, TestFunctions, TrialFunctions, cos, pi, sin)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_vector, create_matrix, set_bc)
import numpy as np
import numpy.typing as npt
import mesh_init
import solver_classes.Navier_stokes_solver
import solver_classes.ODE_solver
from basix.ufl import element, mixed_element
import matplotlib.pyplot as plt
from helper_functions.helper_functions import test_gradient, eval_vector_field, \
    test_gradient_centered_finite_differences, test_gradient_centered_finite_differences_on_rhs_control
import imageio.v2 as imageio
from stokes_helper import solve_stokes

# ----------------------------------------------------------------------------------------------------------------------
experiment_number = 30
np_path = f"test_pipelines/results/Stokes_rhs_control/{experiment_number}/"

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
num_steps = 0
# ----------------------------------------------------------------------------------------------------------------------
# Mesh
# gdim = 2
# mesh, ft, inlet_marker, wall_marker = mesh_init.create_mesh(gdim)
# mesh, ft, inlet_marker, wall_marker, outlet_marker, inlet_coord, right_coord = mesh_init.create_pipe_mesh(gdim)
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 32, 32, dolfinx.mesh.CellType.quadrilateral)
# ----------------------------------------------------------------------------------------------------------------------
# init q
U_el = element("CG", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
P_el = element("CG", mesh.basix_cell(), 1)
W_el = mixed_element([U_el, P_el])
W = functionspace(mesh, W_el)
U, _ = W.sub(0).collapse()
dx = Measure("dx", mesh)
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

# dx = ufl.Measure('dx', domain=mesh)

u_r, p_r = TrialFunctions(W)
v_r, pr_r = TestFunctions(W)

u_r_adj, p_r_adj = TrialFunctions(W)
v_r_adj, pr_r_adj = TestFunctions(W)


def nonslip(x):
    values = np.zeros((2, x.shape[1]))
    return values


u_nonslip = Function(U)
u_nonslip.interpolate(nonslip)

tdim = mesh.topology.dim
fdim = tdim - 1
mesh.topology.create_connectivity(fdim, tdim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)

boundary_dofs = dolfinx.fem.locate_dofs_topological((W.sub(0), U), fdim, boundary_facets)
bc = dolfinx.fem.dirichletbc(u_nonslip, boundary_dofs, W.sub(0))


def ud_in(x):
    values = np.zeros((2, x.shape[1]))
    values[0, :] = 1
    values[1, :] = 1
    return values


udp = Function(W)
ud, _ = udp.split()
ud.interpolate(ud_in)


def f_in(x):
    values = np.zeros((2, x.shape[1]))
    values[0, :] = x[0]
    values[1, :] = x[0]
    return values


fp = Function(W)
f, _ = fp.split()
f.interpolate(f_in)

a1 = vis * inner(grad(u_r), grad(v_r)) * dx
b = inner(p_r, div(v_r)) * dx
div_ = inner(pr_r, div(u_r)) * dx
rhs_ = inner(f, v_r) * dx
F = a1 + div_ + b

problem = dolfinx.fem.petsc.LinearProblem(F, rhs_, bcs=[bc])
w_s = problem.solve()
u_s, p_s = w_s.split()

a1_adj = vis * inner(grad(u_r_adj), grad(v_r_adj)) * dx
b_adj = inner(p_r_adj, div(v_r_adj)) * dx
div_adj = inner(pr_r_adj, div(u_r_adj)) * dx
rhs_adj = inner(u_s - ud, v_r_adj) * dx
F_adj = a1_adj + div_adj + b_adj

problem_adj = dolfinx.fem.petsc.LinearProblem(F_adj, rhs_adj, bcs=[bc])
w_s_adj = problem_adj.solve()
u_s_adj, p_s_adj = w_s_adj.split()

# ----------------------------------------------------------------------------------------------------------------------
i = 0
test_gradient_centered_finite_differences_on_rhs_control(u_s_adj, f, i, W, alpha, np_path, mesh, F, v_r, bc, u_s, ud)
div_u = form(div(u_s)**2 * dx)
divs_u.append(assemble_scalar(div_u))
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

# write vector field for paraview
# vtx_u = dolfinx.io.VTXWriter(mesh.comm, np_path + f"{experiment_number}_u.bp", w_r.sub(0).collapse(), engine="BP4")
# vtx_u.write(0.0)
# vtx_u.close()
