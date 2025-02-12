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
                 sqrt, transpose, tr, TestFunctions, TrialFunctions)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_vector, create_matrix, set_bc)
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
experiment_number = 4
np_path = f"test_pipelines/results/Stokes/{experiment_number}/"

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
gmsh.initialize()
gdim = 2
# mesh, ft, inlet_marker, wall_marker = mesh_init.create_mesh(gdim)
mesh, ft, inlet_marker, wall_marker, outlet_marker, inlet_coord, right_coord = mesh_init.create_pipe_mesh(gdim)

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

fdim = mesh.topology.dim - 1
U_el = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
P_el = element("Lagrange", mesh.basix_cell(), 1)
W_el = mixed_element([U_el, P_el])
W = functionspace(mesh, W_el)
U, _ = W.sub(0).collapse()

dx = ufl.Measure('dx', domain=mesh)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)

u_r, p_r = TrialFunctions(W)
v_r, pr_r = TestFunctions(W)

u_r_adj, p_r_adj = TrialFunctions(W)
v_r_adj, pr_r_adj = TestFunctions(W)


def nonslip(x):
    values = np.zeros((2, x.shape[1]))
    return values


u_nonslip = Function(U)
u_nonslip.interpolate(nonslip)

wall_dofs = locate_dofs_topological((W.sub(0), U), fdim, ft.find(wall_marker))
bcu_wall = dirichletbc(u_nonslip, wall_dofs, W.sub(0))
bcu = [bcu_wall]


def set_stokes_state(q):
    f = Constant(mesh, PETSc.ScalarType((0, 0)))
    a = vis * inner(grad(u_r), grad(v_r)) * dx
    b = inner(p_r, div(v_r)) * dx
    div_ = inner(pr_r, div(u_r)) * dx
    f_ = inner(f, v_r) * dx + inner(q, v_r) * ds(inlet_marker)
    F = a + div_ - b  # - extra_bt
    a = form(F)
    L1 = form(f_)
    A = assemble_matrix(a, bcs=bcu)
    A.assemble()
    b1 = create_vector(L1)
    return A, b1, a, L1


def set_adjoint_stokes(u):
    f = Constant(mesh, PETSc.ScalarType((0, 0)))
    a = vis * inner(grad(u_r_adj), grad(v_r_adj)) * dx
    b = inner(p_r_adj, div(v_r_adj)) * dx
    div_ = inner(pr_r_adj, div(u_r_adj)) * dx
    f_ = inner(u, v_r_adj) * dx
    F = a + div_ - b  # - extra_bt
    a = form(F)
    L1 = form(f_)
    A = assemble_matrix(a, bcs=bcu)
    A.assemble()
    b1 = create_vector(L1)
    return A, b1, a, L1


def solve_stokes_step(q):
    A, b, a, L = set_stokes_state(q)
    w_s = Function(W)
    solver1 = PETSc.KSP().create(mesh.comm)
    solver1.setOperators(A)
    solver1.setType(PETSc.KSP.Type.BCGS)
    pc1 = solver1.getPC()
    pc1.setType(PETSc.PC.Type.JACOBI)

    A.zeroEntries()
    assemble_matrix(A, a, bcs=bcu)
    A.assemble()
    with b.localForm() as loc:
        loc.set(0)
    assemble_vector(b, L)
    apply_lifting(b, [a], [bcu])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcu)
    solver1.solve(b, w_s.x.petsc_vec)
    w_s.x.scatter_forward()
    return w_s


def solve_adoint_stokes_step(u):
    A, b, a, L = set_adjoint_stokes(u)
    w_s = Function(W)
    solver1 = PETSc.KSP().create(mesh.comm)
    solver1.setOperators(A)
    solver1.setType(PETSc.KSP.Type.BCGS)
    pc1 = solver1.getPC()
    pc1.setType(PETSc.PC.Type.JACOBI)

    A.zeroEntries()
    assemble_matrix(A, a, bcs=bcu)
    A.assemble()
    with b.localForm() as loc:
        loc.set(0)
    assemble_vector(b, L)
    apply_lifting(b, [a], [bcu])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcu)
    solver1.solve(b, w_s.x.petsc_vec)
    w_s.x.scatter_forward()
    return w_s


# ----------------------------------------------------------------------------------------------------------------------
dx_ = ufl.Measure('dx', domain=mesh)
# ----------------------------------------------------------------------------------------------------------------------
# optimization loop
for i in range(num_steps):
    w_r = solve_stokes_step(q)
    u, p = w_r.split()
    w_adj = solve_adoint_stokes_step(u)
    z, _ = w_adj.split()
    q.x.array[:] = q.x.array - mu * (alpha * q.x.array - z.x.array)

    if i == 0:
        # vtx_u_init = dolfinx.io.VTXWriter(mesh.comm, np_path + f"{experiment_number}_u_init.bp", w_r.sub(0).collapse(),
        #                                   engine="BP4")
        # vtx_u_init.write(0.0)
        # vtx_u_init.close()
        test_gradient(z, qp.sub(0), u, None, i, None, ft, W, alpha, inlet_marker, None, h, solve_stokes_step,
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
