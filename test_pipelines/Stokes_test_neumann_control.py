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
                 sqrt, transpose, tr, TestFunctions, TrialFunctions, cos, pi, sin, )
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
    test_gradient_centered_finite_differences
import imageio.v2 as imageio
from stokes_helper import solve_stokes

# ----------------------------------------------------------------------------------------------------------------------
experiment_number = 16
np_path = f"test_pipelines/results/Stokes_neumann_control/{experiment_number}/"

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
num_steps = 1
# ----------------------------------------------------------------------------------------------------------------------
# Mesh
gdim = 2
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 48, 48, dolfinx.mesh.CellType.triangle)
# ----------------------------------------------------------------------------------------------------------------------
# init q
U_el = element("CG", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
P_el = element("CG", mesh.basix_cell(), 1)
W_el = mixed_element([U_el, P_el])
W = functionspace(mesh, W_el)
qp = Function(W)
q, _ = qp.split()
U, _ = W.sub(0).collapse()

top, bottom, left, right = False, False, False, True


def boundary_values(x):
    values = np.zeros((2, x.shape[1]))
    if left:
        values[0, :] = np.where(np.isclose(x[0, :], 0.0),
                                x[1]*(1-x[1]),
                                0.0)
                                #np.logical_and(np.where(np.isclose(x[0, :], 0.5_2b), 10000, 0.0),np.where(np.isclose(x[1, :], 0.5_2b), 10000, 0.0)))
        values[1, :] = np.where(np.isclose(x[0, :], 0.0),
                                0,
                                0.0)
    if right:
        values[0, :] = np.where(np.isclose(x[0, :], 1.0),
                                x[1],0.0)
                                #np.logical_and(np.where(np.isclose(x[0, :], 0.5_2b), 10000, 0.0),np.where(np.isclose(x[1, :], 0.5_2b), 10000, 0.0)))
        values[1, :] = np.where(np.isclose(x[0, :], 1.0),
                                x[1],
                                0.0)
    if top:
        values[0, :] = np.where(np.isclose(x[1, :], 1.0),
                                x[0],
                                0.0)
                                #np.logical_and(np.where(np.isclose(x[0, :], 0.5_2b), 10000, 0.0),np.where(np.isclose(x[1, :], 0.5_2b), 10000, 0.0)))
        values[1, :] = np.where(np.isclose(x[1, :], 1.0),
                                x[0],
                                0.0)
    if bottom:
        values[0, :] = np.where(np.isclose(x[1, :], 0.0),
                                x[0],
                                0.0)
                                #np.logical_and(np.where(np.isclose(x[0, :], 0.5_2b), 10000, 0.0),np.where(np.isclose(x[1, :], 0.5_2b), 10000, 0.0)))
        values[1, :] = np.where(np.isclose(x[1, :], 0.0),
                                x[0],
                                0.0)
    return values


q.interpolate(boundary_values)

# print("write p")
# vtx_p = dolfinx.io.VTXWriter(mesh.comm, np_path + f"{experiment_number}_q.bp", qp.sub(0).collapse(), engine="BP4")
# vtx_p.write(0.0)
# vtx_p.close()


def ud_in(x):
    values = np.zeros((2, x.shape[1]))
    values[0, :] = 1
    values[1, :] = 1
    return values


udp = Function(W)
ud, _ = udp.split()
ud.interpolate(ud_in)
# ----------------------------------------------------------------------------------------------------------------------
tdim = mesh.topology.dim
fdim = tdim - 1
mesh.topology.create_connectivity(fdim, tdim)


# boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)

facets_bottom = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[1], 0.0))
facets_top = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[1],1.0))
facets_right = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[0],1.0))
facets_left = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[0],0.0))

boundary_dofs_top = dolfinx.fem.locate_dofs_topological((W.sub(0), U), fdim, facets_top)
boundary_dofs_bottom = dolfinx.fem.locate_dofs_topological((W.sub(0), U), fdim, facets_bottom)
boundary_dofs_right = dolfinx.fem.locate_dofs_topological((W.sub(0), U), fdim, facets_right)
boundary_dofs_left = dolfinx.fem.locate_dofs_topological((W.sub(0), U), fdim, facets_left)

def nonslip(x):
    values = np.zeros((2, x.shape[1]))
    return values


u_nonslip = Function(U)
u_nonslip.interpolate(nonslip)

bcu_wall_top = dirichletbc(u_nonslip, boundary_dofs_top, W.sub(0))
bcu_wall_bottom = dirichletbc(u_nonslip, boundary_dofs_bottom, W.sub(0))
bcu_wall_right = dirichletbc(u_nonslip, boundary_dofs_right, W.sub(0))
bcu_wall_left = dirichletbc(u_nonslip, boundary_dofs_left, W.sub(0))
bc_array = [bcu_wall_left, bcu_wall_right, bcu_wall_bottom, bcu_wall_top]

boundary_marker_id = [1,1,1,1]
bcu = []
for ir, boundary_bool in enumerate([left, right, bottom, top]):
    if boundary_bool:
        boundary_marker_id[ir] = 2
    else:
        bcu.append(bc_array[ir])


boundaries = [(boundary_marker_id[0], lambda x: np.isclose(x[0], 0)),
              (boundary_marker_id[1], lambda x: np.isclose(x[0], 1.0)),
              (boundary_marker_id[2], lambda x: np.isclose(x[1], 0)),
              (boundary_marker_id[3], lambda x: np.isclose(x[1], 1.0))]

facet_indices, facet_markers = [], []
for (marker, locator) in boundaries:
    facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = dolfinx.mesh.meshtags(mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

dx = ufl.Measure('dx', domain=mesh)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tag)

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
u_r, p_r = TrialFunctions(W)
v_r, pr_r = TestFunctions(W)

u_r_adj, p_r_adj = TrialFunctions(W)
v_r_adj, pr_r_adj = TestFunctions(W)

f = Constant(mesh, PETSc.ScalarType((0, 0)))
a1 = vis * inner(grad(u_r), grad(v_r)) * dx
b = p_r*div(v_r) * dx
div_ = pr_r*div(u_r) * dx
rhs_ = inner(q, v_r) * ds(2)
F = a1 + div_ + b

problem = dolfinx.fem.petsc.LinearProblem(F, rhs_, bcs=bcu,petsc_options={
        "ksp_type": "preonly",
        #"pc_type": "lu",
        #"pc_factor_mat_solver_type": "superlu_dist",
    })
w_s = problem.solve()
u_s, p_s = w_s.split()

a1_adj = vis * inner(grad(u_r_adj), grad(v_r_adj)) * dx
b_adj = inner(p_r_adj, div(v_r_adj)) * dx
div_adj = inner(pr_r_adj, div(u_r_adj)) * dx
rhs_adj = inner(u_s - ud, v_r_adj) * dx
F_adj = a1_adj + div_adj + b_adj

problem_adj = dolfinx.fem.petsc.LinearProblem(F_adj, rhs_adj, bcs=bcu,petsc_options={
        "ksp_type": "preonly",
        #"pc_type": "lu",
        #"pc_factor_mat_solver_type": "superlu_dist",
    })
w_s_adj = problem_adj.solve()
u_s_adj, p_s_adj = w_s_adj.split()
# ----------------------------------------------------------------------------------------------------------------------
i = 0
test_gradient_centered_finite_differences(u_s_adj, q, i, ds, W, alpha, np_path, mesh, F, v_r , bcu, ud, u_s, top, bottom, left, right)

div_u = form(inner(div(u_s), div(u_s)) * dx)
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


# write vector field for paraview
vtx_u = dolfinx.io.VTXWriter(mesh.comm, np_path + f"{experiment_number}_u.bp", w_s.sub(0).collapse(), engine="BP4")
vtx_u.write(0.0)
vtx_u.close()
