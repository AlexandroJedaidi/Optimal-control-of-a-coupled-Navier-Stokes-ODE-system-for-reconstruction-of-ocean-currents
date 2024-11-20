import json
import gmsh
import dolfinx
import mesh_init
import ufl
from ufl import (FacetNormal, Identity, Measure, TestFunction, TrialFunction,
                 as_vector, div, dot, inner, lhs, grad, nabla_grad, rhs, sym, system, SpatialCoordinate, inv,
                 sqrt, transpose, tr)
from dolfinx.fem import (Constant, Function, functionspace,
                         assemble_scalar, dirichletbc, form, locate_dofs_topological, set_bc)
from petsc4py import PETSc
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
from scipy.integrate import odeint
import numpy as np


class ODE:

    def __init__(self, mesh, ft, inlet_marker, wall_marker, mesh_t):
        self.X = None
        self.X_adj = None
        self.lam_1 = None
        self.x = None
        self.lam_2 = None
        self.lam_2_0 = None
        self.x0 = None
        self.u0 = None
        self.u_d = None
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

    def set_function_spaces(self):
        self.X = dolfinx.fem.functionspace(self.mesh_t, ("Lagrange", 2, (self.mesh.geometry.dim,)))
        self.X_adj = self.X.clone()

        self.U = dolfinx.fem.functionspace(self.mesh, ("Lagrange", 2, (self.mesh.geometry.dim,)))
        self.U_adj = self.U.clone()

    def set_functions(self):
        self.lam_1 = TrialFunction(self.U_adj)

        self.x = [Function(self.X) for _ in range(self.K)]
        self.lam_2 = [Function(self.X_adj) for _ in range(self.K)]

        self.x0 = [[0.0 + i, 0.0 + i] for i in range(self.K)]
        self.lam_2_0 = [[0.0, 0.0] for i in range(self.K)]


    def ode_solving_step(self, u_dolf):

        def u_scp(t, x):
            point = np.array([x[0], x[1], 0])
            bb_tree = dolfinx.geometry.bb_tree(self.mesh, self.mesh.topology.dim)
            cells = []
            points_on_proc = []
            # Find cells whose bounding-box collide with the the points
            cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, point)
            # Choose one of the cells that contains the point
            colliding_cells = dolfinx.geometry.compute_colliding_cells(self.mesh, cell_candidates, point)
            if len(colliding_cells.links(0)) > 0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(0)[0])
            points_on_proc = np.array(points_on_proc, dtype=np.float64)
            u_scp = u_dolf.eval(points_on_proc, cells)

            return u_scp

        #u_scp = lambda t, x: u_dolf(x) # TODO: to np
        self.sol = []
        for k in range(self.K):
            sol_np = odeint(u_scp, self.x0[k], np.linspace(self.t0, self.T))
            sol_dolfinx = ...# TODO: to dolfinx
            self.sol.append(sol_dolfinx)
        return self.sol

    def adjoint_ode_solving_step(self, u_dolf, u_dk, x_k):
        self.adj_sol_dolfinx = []
        for k in range(self.K):
            grad_u_np = ufl.grad(u_dolf(x_k[k])) # TODO: to np
            u_scp = lambda t, lam_2: - grad_u_np @ (u_dolf(x_k[k]) - u_dk[k] + lam_2)
            adj_sol_np = odeint(u_scp, self.lam_2_0[k], np.linspace(-self.T, -self.t0))
            adj_sol_np = adj_sol_np[::-1]
            adj_sol_dolfinx = ... # TODO: to dolfinx
            self.sol.append(adj_sol_dolfinx)
        return self.adj_sol_dolfinx

