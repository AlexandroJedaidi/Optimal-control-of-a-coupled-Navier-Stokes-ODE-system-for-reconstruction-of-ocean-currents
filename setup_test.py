import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import gmsh
import dolfinx

import numpy as np
import matplotlib.pyplot as plt
import tqdm.autonotebook
from mpi4py import MPI
from petsc4py import PETSc
from basix.ufl import element

from dolfinx.cpp.mesh import to_type, cell_entity_type
from dolfinx.fem import (Constant, Function, functionspace,
                         assemble_scalar, dirichletbc, form, locate_dofs_topological, set_bc)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_vector, create_matrix, set_bc)
from dolfinx.graph import adjacencylist
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from dolfinx.io import (VTXWriter, distribute_entity_data, gmshio)
from dolfinx.mesh import create_mesh, meshtags_from_entities
from ufl import (FacetNormal, Identity, Measure, TestFunction, TrialFunction,
                 as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym, system, SpatialCoordinate, inv,
                 sqrt, transpose, tr)
import sys

sys.path.append('models/FENE-P/')
import fene_p
import mesh_init

gmsh.initialize()
gdim = 2
mesh, ft, inlet_marker, wall_marker, outlet_marker, obstacle_marker = mesh_init.create_mesh(gdim, False)

# ---------------------------------------------------------------------------------------------------------------------
experiment_number = 1
experiment_path = f'results/experiments/{experiment_number}'
#os.mkdir(experiment_path)
np_path = f'results/experiments/{experiment_number}/arrays'
plot_path = f"results/experiments/{experiment_number}/plots"
#os.mkdir(np_path)
#os.mkdir(plot_path)
# ---------------------------------------------------------------------------------------------------------------------
# discretization parameters
t = 0
T = 1.0  # Final time
dt = 1 / (100)  # Time step size
num_steps = int(T / dt)
k = Constant(mesh, PETSc.ScalarType(dt))

# flow properties for navier stokes2
U_n = 1  # mean inlet velocity
L_n = 0.1  # characteristic length
rho_n = 1.0  # density
vs_n = 0.001  # fluid visc.

# flow properties for fokker planck
vp_n = 0.0035  # polymer visc.
b = 100  # dumbbell length
lambd = 0.03
Wi = lambd*U_n/L_n  # 0.03  # Weissenberg number
alpha = 0.01  # extra diffusion scale

# mixture properties0
vis = vs_n + vp_n  # total visc.
# b_n = vs_n / vis  # solvent ratio
b_n = 0.85
# Re_n = (U_n * L_n) / vis  # reynolds number
Re_n = 100

# doflinx parameter initialization
beta = Constant(mesh, PETSc.ScalarType(b_n))
mu = Constant(mesh, PETSc.ScalarType(vis))
vp = Constant(mesh, PETSc.ScalarType(vp_n))
vs = Constant(mesh, PETSc.ScalarType(vs_n))
Re = Constant(mesh, PETSc.ScalarType(Re_n))
rho = Constant(mesh, PETSc.ScalarType(rho_n))
U = Constant(mesh, PETSc.ScalarType(U_n))
L = Constant(mesh, PETSc.ScalarType(L_n))
# ---------------------------------------------------------------------------------------------------------------------
# parameter output
with open(np_path + "variables.txt", "w") as text_file:
    text_file.write("fluid viscosity: %s \n" % vs_n)
    text_file.write("polymer viscosity: %s \n" % vp_n)
    text_file.write("Reynolds Number: %s \n" % Re_n)
    text_file.write("Weissenberg Number: %s \n" % Wi)
    text_file.write("Max Extension: %s \n" % b)
    text_file.write("dt: %s \n" % dt)
    text_file.write("T: %s \n" % T)
    text_file.write("beta: %s" % b_n)

# ---------------------------------------------------------------------------------------------------------------------
# navier stokes function spaces
v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2, shape=(mesh.geometry.dim,))
s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
V = functionspace(mesh, v_cg2)
Q = functionspace(mesh, s_cg1)

fdim = mesh.topology.dim - 1  # dimension


# ---------------------------------------------------------------------------------------------------------------------
# System boundary conditions
class InletVelocity():
    def __init__(self, t):
        self.t = t

    def __call__(self, x):
        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        if self.t < 2:  # ramp up time t=2
            values[0] = 2 * 1.5 * (1 - np.cos(self.t * np.pi / 2)) * x[1] * (0.41 - x[1]) / (0.41 ** 2)
        else:
            values[0] = 6 * x[1] * (0.41 - x[1]) / (0.41 ** 2)
        return values

# Inlet
u_inlet = Function(V)
inlet_velocity = InletVelocity(t)
u_inlet.interpolate(inlet_velocity)
bcu_inflow = dirichletbc(u_inlet, locate_dofs_topological(V, fdim, ft.find(inlet_marker)))
# Walls
u_nonslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
bcu_walls = dirichletbc(u_nonslip, locate_dofs_topological(V, fdim, ft.find(wall_marker)), V)

# Obstacle
bcu_obstacle = dirichletbc(u_nonslip, locate_dofs_topological(V, fdim, ft.find(obstacle_marker)), V)
bcu = [bcu_inflow, bcu_obstacle, bcu_walls]

# Outlet
bcp_outlet = dirichletbc(PETSc.ScalarType(0), locate_dofs_topological(Q, fdim, ft.find(outlet_marker)), Q)
bcp = [bcp_outlet]
# ---------------------------------------------------------------------------------------------------------------------
# define solution and test functions for navier stokes
u = TrialFunction(V)
v = TestFunction(V)
u_ = Function(V)
u_.name = "u"
u_s = Function(V)
u_n = Function(V)
u_n1 = Function(V)
p = TrialFunction(Q)
q = TestFunction(Q)
p_ = Function(Q)
p_.name = "p"
phi = Function(Q)
# ---------------------------------------------------------------------------------------------------------------------
# FENE-P Initialization
# FENE-P function spaces solution and test functions
S, sigma, phi_tf = fene_p.function_space(mesh)
x = SpatialCoordinate(mesh)
# Fene-P boundary conditions
bc = fene_p.boundary_conditions(mesh, S, x)
# Fokker Planck solution data allocation
sigma_n, sigma_11_solution_data, sigma_12_solution_data, sigma_21_solution_data, sigma_22_solution_data, time_values_data = fene_p.solution_initialization(
    num_steps, S)
# ---------------------------------------------------------------------------------------------------------------------
# variational/weak formulation of navier stokes
n = FacetNormal(mesh)
f = Constant(mesh, PETSc.ScalarType((0, 0)))
# div_tau = (beta*(b+2)/b)/Wi * tr(((fene_p.A(sigma, b)) * sigma - Identity(2))*transpose(grad(v)))
div_tau = (1-beta)/(Re*Wi) * dot(div((fene_p.A(sigma, b)) * sigma - Identity(2)), v)
F1 = 1 / k * dot(u - u_n, v) * dx
F1 += inner(dot(1.5 * u_n - 0.5 * u_n1, 0.5 * nabla_grad(u + u_n)), v) * dx
F1 += beta/Re * 0.5 * inner(grad(u + u_n), grad(v)) * dx - dot(p_, div(v)) * dx
# F1 += dot(p_ * n, v) * ds - dot(mu * nabla_grad(0.5 * (u_n + u)) * n, v) * ds # = 0 for do-nothing BC
F1 -= div_tau * dx  # extra stress
F1 -= dot(f, v) * dx

# assemble matrices for the linear system
a1 = form(lhs(F1))
L1 = form(rhs(F1))
A1 = create_matrix(a1)
b1 = create_vector(L1)

a2 = form(dot(grad(p), grad(q)) * dx)
L2 = form(-rho / k * dot(div(u_s), q) * dx)
A2 = assemble_matrix(a2, bcs=bcp)
A2.assemble()
b2 = create_vector(L2)

a3 = form(rho * dot(u, v) * dx)
L3 = form(rho * dot(u_s, v) * dx - k * dot(nabla_grad(phi), v) * dx)
A3 = assemble_matrix(a3)
A3.assemble()
b3 = create_vector(L3)
# ---------------------------------------------------------------------------------------------------------------------
# navier stokes solver initialization
# Solver for step 1
solver1 = PETSc.KSP().create(mesh.comm)
solver1.setOperators(A1)
solver1.setType(PETSc.KSP.Type.BCGS)
pc1 = solver1.getPC()
pc1.setType(PETSc.PC.Type.JACOBI)

# Solver for step 2
solver2 = PETSc.KSP().create(mesh.comm)
solver2.setOperators(A2)
solver2.setType(PETSc.KSP.Type.MINRES)
pc2 = solver2.getPC()
pc2.setType(PETSc.PC.Type.HYPRE)
pc2.setHYPREType("boomeramg")

# Solver for step 3
solver3 = PETSc.KSP().create(mesh.comm)
solver3.setOperators(A3)
solver3.setType(PETSc.KSP.Type.CG)
pc3 = solver3.getPC()
pc3.setType(PETSc.PC.Type.SOR)
# ---------------------------------------------------------------------------------------------------------------------
# physical measurements
n = -FacetNormal(mesh)  # Normal pointing out of obstacle
dObs = Measure("ds", domain=mesh, subdomain_data=ft, subdomain_id=obstacle_marker)
dout = Measure("ds", domain=mesh, subdomain_data=ft, subdomain_id=outlet_marker)
u_t = inner(as_vector((n[1], -n[0])), u_)
drag = form(2/0.1 * (0.1/Re * inner(grad(u_t*U_n), n) * n[1] - p_ * n[0]) * dObs)
lift = form(-2/0.1 * (0.1/Re * inner(grad(u_t*U_n), n) * n[0] + p_ * n[1]) * dObs)
if mesh.comm.rank == 0:
    C_D = np.zeros(num_steps, dtype=PETSc.ScalarType)
    C_L = np.zeros(num_steps, dtype=PETSc.ScalarType)
    C_T = np.zeros(num_steps, dtype=PETSc.ScalarType)
    t_u = np.zeros(num_steps, dtype=np.float64)
    t_p = np.zeros(num_steps, dtype=np.float64)

if mesh.comm.rank == 0:
    t_u = np.zeros(num_steps, dtype=np.float64)
    t_p = np.zeros(num_steps, dtype=np.float64)

tree = bb_tree(mesh, mesh.geometry.dim)
points = np.array([[0.15, 0.2, 0], [0.25, 0.2, 0]])
cell_candidates = compute_collisions_points(tree, points)
colliding_cells = compute_colliding_cells(mesh, cell_candidates, points)
front_cells = colliding_cells.links(0)
back_cells = colliding_cells.links(1)
if mesh.comm.rank == 0:
    p_diff = np.zeros(num_steps, dtype=PETSc.ScalarType)

u_magnitude = []  # flow speed list
cond = []
# ---------------------------------------------------------------------------------------------------------------------
progress = tqdm.autonotebook.tqdm(desc="Solving PDE", total=num_steps)  # progress bar
# ---------------------------------------------------------------------------------------------------------------------
# iteration of given time interval
for i in range(num_steps):
    progress.update(1)
    # Update current time step
    t += dt
    # Update inlet velocity
    inlet_velocity.t = t
    u_inlet.interpolate(inlet_velocity)
    # ---------------------------------------------------------------------------------------------------------------------
    # Step 1: Navier stokes solving
    # Step 1.1: Tentative velocity step
    A1.zeroEntries()
    assemble_matrix(A1, a1, bcs=bcu)
    A1.assemble()
    with b1.localForm() as loc:
        loc.set(0)
    assemble_vector(b1, L1)
    apply_lifting(b1, [a1], [bcu])
    b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b1, bcu)
    solver1.solve(b1, u_s.x.petsc_vec)
    u_s.x.scatter_forward()

    # Step 1.2: Pressure correction step
    with b2.localForm() as loc:
        loc.set(0)
    assemble_vector(b2, L2)
    apply_lifting(b2, [a2], [bcp])
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b2, bcp)
    solver2.solve(b2, phi.x.petsc_vec)
    phi.x.scatter_forward()

    p_.x.petsc_vec.axpy(1, phi.x.petsc_vec)
    p_.x.scatter_forward()

    # Step 1.3: Velocity correction step
    with b3.localForm() as loc:
        loc.set(0)
    assemble_vector(b3, L3)
    b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    solver3.solve(b3, u_.x.petsc_vec)
    u_.x.scatter_forward()
    # ---------------------------------------------------------------------------------------------------------------------
    # Step 2: FENE-P solving
    crash = fene_p.solve(sigma, sigma_n, dt, u_[0], u_[1], bc, phi_tf, b, Wi, alpha,cond)
    if crash:
        print(f"FENE-P pipeline crashed at t={t}!")
        break
    # saving current sigma solutions to lists
    fene_p.save_solutions(sigma, sigma_11_solution_data, sigma_12_solution_data, sigma_21_solution_data,
                          sigma_22_solution_data, time_values_data, i, t)
    # update next time step
    sigma.x.scatter_forward()
    sigma_n.x.array[:] = sigma.x.array
    # ---------------------------------------------------------------------------------------------------------------------
    # Update variable with solution form this time step
    with u_.x.petsc_vec.localForm() as loc_, u_n.x.petsc_vec.localForm() as loc_n, u_n1.x.petsc_vec.localForm() as loc_n1:
        loc_n.copy(loc_n1)
        loc_.copy(loc_n)

    # Compute physical quantities
    # For this to work in parallel, we gather contributions from all processors
    # to processor zero and sum the contributions.
    drag_coeff = mesh.comm.gather(assemble_scalar(drag), root=0)
    lift_coeff = mesh.comm.gather(assemble_scalar(lift), root=0)
    #tau_coeff = mesh.comm.gather(assemble_scalar(form(div_tau * dx)), root=0)

    p_front = None
    if len(front_cells) > 0:
        p_front = p_.eval(points[0], front_cells[:1])
    p_front = mesh.comm.gather(p_front, root=0)
    p_back = None
    if len(back_cells) > 0:
        p_back = p_.eval(points[1], back_cells[:1])
    p_back = mesh.comm.gather(p_back, root=0)
    if mesh.comm.rank == 0:
        t_u[i] = t
        t_p[i] = t - dt / 2
        C_D[i] = sum(drag_coeff)
        C_L[i] = sum(lift_coeff)
        #C_T[i] = sum(tau_coeff)
        # Choose first pressure that is found from the different processors
        for pressure in p_front:
            if pressure is not None:
                p_diff[i] = pressure[0]
                break
        for pressure in p_back:
            if pressure is not None:
                p_diff[i] -= pressure[0]
                break
    # ---------------------------------------------------------------------------------------------------------------------
    # save current magnitude
    u_len = int(u_.x.array.shape[0] / 2)
    u_mag_gen = (np.sqrt((u_.x.array[2 * k]) ** 2 + (u_.x.array[2 * k + 1]) ** 2) for k in range(u_len))
    u_magnitude.append(list(u_mag_gen))
# ---------------------------------------------------------------------------------------------------------------------
# storing all solution arrays
# navier stokes solution
with open(np_path + 'u_time.npy', 'wb') as f:
    np.save(f, np.array(t_u))
with open(np_path + 'p_time.npy', 'wb') as f:
    np.save(f, np.array(t_p))

with open(np_path + 'u_mag.npy', 'wb') as f:
    np.save(f, np.array(u_magnitude))
with open(np_path + 'F_cond.npy', 'wb') as f:
    np.save(f, np.array(cond))

# fokker plank solution
with open(np_path + 'sigma11.npy', 'wb') as f:
    np.save(f, np.array(sigma_11_solution_data))
with open(np_path + 'sigma12.npy', 'wb') as f:
    np.save(f, np.array(sigma_12_solution_data))
with open(np_path + 'sigma21.npy', 'wb') as f:
    np.save(f, np.array(sigma_21_solution_data))
with open(np_path + 'sigma22.npy', 'wb') as f:
    np.save(f, np.array(sigma_22_solution_data))

# physical quantities
with open(np_path + 'drag_coeff.npy', 'wb') as f:
    np.save(f, np.array(C_D))
with open(np_path + 'lift_coeff.npy', 'wb') as f:
    np.save(f, np.array(C_L))
with open(np_path + 'pressure_coeff.npy', 'wb') as f:
    np.save(f, np.array(p_diff))
# extra stress
# with open(np_path + 'tau_array.npy', 'wb') as f:
#     np.save(f, np.array(C_T))

# ---------------------------------------------------------------------------------------------------------------------
# display physical quantities
if mesh.comm.rank == 0:
    if not os.path.exists("results/figures"):
        os.mkdir("results/figures")
    num_velocity_dofs = V.dofmap.index_map_bs * V.dofmap.index_map.size_global
    num_pressure_dofs = Q.dofmap.index_map_bs * V.dofmap.index_map.size_global

    fig = plt.figure(figsize=(25, 8))
    l1 = plt.plot(t_u, C_D, label=r"FEniCSx  ({0:d} dofs)".format(num_velocity_dofs + num_pressure_dofs), linewidth=2)
    plt.title("Drag coefficient")
    plt.grid()
    plt.legend()
    plt.savefig(plot_path + "drag_comparison.png")

    fig = plt.figure(figsize=(25, 8))
    l1 = plt.plot(t_u, C_L, label=r"FEniCSx  ({0:d} dofs)".format(
        num_velocity_dofs + num_pressure_dofs), linewidth=2)
    plt.title("Lift coefficient")
    plt.grid()
    plt.legend()
    plt.savefig(plot_path + "lift_comparison.png")

    fig = plt.figure(figsize=(25, 8))
    l1 = plt.plot(t_p, p_diff, label=r"FEniCSx ({0:d} dofs)".format(num_velocity_dofs + num_pressure_dofs), linewidth=2)
    plt.title("Pressure difference")
    plt.grid()
    plt.legend()
    plt.savefig(plot_path + "pressure_comparison.png")

    # fig = plt.figure(figsize=(25, 8))
    # l1 = plt.plot(t_p, C_T, label=r"FEniCSx ({0:d} dofs)".format(num_velocity_dofs + num_pressure_dofs), linewidth=2)
    # plt.title("Summed tau")
    # plt.grid()
    # plt.legend()
    # plt.savefig(plot_path + "tau.png")
# ---------------------------------------------------------------------------------------------------------------------
# storing vtx file of velocity u and pressure p
with dolfinx.io.VTXWriter(MPI.COMM_WORLD, np_path + f"{experiment_number}_{str(b_n)}_pressure.bp", [p_],
                          engine="BP4") as vtx:
    vtx.write(0.0)
with dolfinx.io.VTXWriter(MPI.COMM_WORLD, np_path + f"{experiment_number}_{str(b_n)}_u.bp", [u_], engine="BP4") as vtx:
    vtx.write(0.0)
# ---------------------------------------------------------------------------------------------------------------------