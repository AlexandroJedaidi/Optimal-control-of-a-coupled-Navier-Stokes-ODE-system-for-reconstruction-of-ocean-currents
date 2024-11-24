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
import mesh_init
import solver_classes.Navier_stokes_solver
import solver_classes.ODE_solver
# ----------------------------------------------------------------------------------------------------------------------
# Discretization parameters
with open("parameters.json", "r") as file:
    parameters = json.load(file)
    t0 = parameters["t0"]
    T = parameters["T"]
    dt = parameters["dt"]
    vis = parameters["viscosity"]
    K = parameters["buoy count"]

    #num_steps = int(T / dt)
# ----------------------------------------------------------------------------------------------------------------------
# Mesh
gmsh.initialize()
gdim = 2
mesh, ft, inlet_marker, wall_marker, mesh_t = mesh_init.create_mesh(gdim)
# ----------------------------------------------------------------------------------------------------------------------
NS_instance = solver_classes.Navier_stokes_solver.NavierStokes(mesh, ft, inlet_marker, wall_marker, mesh_t)
ODE_instance = solver_classes.ODE_solver.ODE(mesh, ft, inlet_marker, wall_marker, mesh_t)
num_steps = 1000
for i in range(num_steps):
    q = np.ones(33)
    u = NS_instance.state_solving_step(q)
    x = ODE_instance.ode_solving_step(u)
    lamb_2 = ODE_instance.adjoint_ode_solving_step(u, NS_instance.u_d_dolfin, x)
    lamb_1 = NS_instance.adjoint_state_solving_step(lamb_2, u, x)

    grad_j = q - lamb_1

    q_n = ... #Grad_desc(j, grad_j, q)
