import json
import gmsh
import dolfinx
import mesh_init
import ufl
from ufl import (FacetNormal, Identity, Measure, TestFunction, TrialFunction,
                 as_vector, div, dot, inner, lhs, grad, nabla_grad, rhs, sym, system, SpatialCoordinate, inv,
                 sqrt, transpose, tr, ds)
from dolfinx.fem import (Constant, Function, functionspace,
                         assemble_scalar, dirichletbc, form, locate_dofs_topological, set_bc)
from petsc4py import PETSc
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
import numpy as np
import scipy as sc


class NavierStokes:

    def __init__(self, mesh, ft, inlet_marker, wall_marker, mesh_t):
        self.lam_2 = None
        self.X_adj = None
        self.X = None
        self.bc = None
        self.x = None
        self.ds = None
        self.dt = None
        self.dx = None
        self.u_d_dolfin = None
        self.q = None
        self.F_adj = None
        self.F = None
        self.Q_adj = None
        self.U_adj = None
        self.U = None
        self.Q = None
        self.mesh = mesh
        self.ft = ft
        self.inlet_marker = inlet_marker
        self.wall_marker = wall_marker
        self.mesh_t = mesh_t
        self.fdim = mesh.topology.dim - 1
        with open("parameters.json", "r") as file:
            parameters = json.load(file)
            self.t0 = parameters["t0"]
            self.T = parameters["T"]
            self.h = parameters["dt"]
            self.viscosity = parameters["viscosity"]
            self.K = parameters["buoy count"]

        self.set_function_spaces()
        self.set_functions()
        self.set_boundary_conditions()
        self.set_adjoint_equation()

    def set_function_spaces(self):
        self.Q = dolfinx.fem.functionspace(self.mesh, ("Lagrange", 2, (self.mesh.geometry.dim,)))
        self.U = dolfinx.fem.functionspace(self.mesh, ("Lagrange", 2, (self.mesh.geometry.dim,)))
        self.U_adj = self.U.clone()
        self.Q_adj = self.Q.clone()

        self.X = dolfinx.fem.functionspace(self.mesh_t, ("Lagrange", 2, (self.mesh.geometry.dim,)))
        self.X_adj = self.X.clone()

    def set_boundary_conditions(self):
        #n = FacetNormal(self.mesh)
        u_nonslip = np.array((0,) * self.mesh.geometry.dim, dtype=PETSc.ScalarType)
        self.bc = dirichletbc(u_nonslip, locate_dofs_topological(self.U, self.fdim, self.ft.find(self.wall_marker)), self.U)
        self.dx = ufl.Measure('dx', domain=self.mesh)
        self.dt = ufl.Measure('dx', domain=self.mesh_t)
        self.ds = ufl.Measure("ds", subdomain_data=self.inlet_marker)

    def set_functions(self):
        (self.u, self.v) = (Function(self.U), TestFunction(self.U))

        (self.lam_1, self.delta_u) = (TrialFunction(self.U_adj), TestFunction(self.U_adj))

        self.x = [Function(self.X) for _ in range(self.K)]
        self.lam_2 = [Function(self.X_adj) for _ in range(self.K)]

        self.u_d_dolfin = [Constant(self.mesh, PETSc.ScalarType((0, 0))) for _ in range(self.K)]

    def set_state_equation(self, q):
        f = Constant(self.mesh, PETSc.ScalarType((0, 0)))
        F1 = self.viscosity * inner(grad(self.u), grad(self.v)) * self.dx
        F1 += inner(dot(self.u, nabla_grad(self.u)), self.v) * self.dx
        F1 -= inner(q, self.v) * ds    #can be done over whole Gamma, since v = 0 on Gamma_2
        F1 -= inner(f, self.v) * self.dx
        return F1

    def set_adjoint_equation(self):
        F2 = self.viscosity * inner(grad(self.lam_1), grad(self.delta_u)) * self.dx
        F2 += inner(self.lam_1, dot(self.delta_u, nabla_grad(self.u))) * self.dx
        F2 += inner(self.lam_1, dot(self.u, nabla_grad(self.delta_u))) * self.dx
        for i in range(0, self.K):
            F2 += inner(self.lam_2[i], self.delta_u(self.x[i])) * self.dt
        for i in range(0, self.K):
            F2 += inner(self.u - self.u_d_dolfin[i], self.delta_u(self.x[i])) * self.dt
        self.F_adj = F2

    def state_solving_step(self, q):
        q_dolf = Function(self.Q)
        f = sc.interpolate.interpolate(q_dolf, self.u)
        q_dolf = q # TODO
        F = self.set_state_equation(q_dolf)
        problem = dolfinx.fem.petsc.NonlinearProblem(F, self.u, bcs=[self.bc])
        solver = NewtonSolver(MPI.COMM_WORLD, problem)
        solver.convergence_criterion = "incremental"
        solver.report = True
        solver.rtol = 1e-4
        solver.max_it = 1000
        n, converged = solver.solve(self.u)
        return self.u

    def adjoint_state_solving_step(self, lam_2, u, x):
        self.lam_2 = lam_2
        self.u = u
        self.x = x
        adj_problem = dolfinx.fem.petsc.NonlinearProblem(self.F_adj, self.lam_1)
        adj_solver = NewtonSolver(MPI.COMM_WORLD, adj_problem)
        adj_solver.convergence_criterion = "incremental"
        adj_solver.report = True
        adj_solver.rtol = 1e-4
        adj_solver.max_it = 1000
        n, converged = adj_solver.solve(self.lam_1)
        return self.lam_1

