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
import scipy as sc
import math


def reconstruct_mesh_points(axis):
    ind = np.unravel_index(np.argsort(axis, axis=None), axis.shape)
    x_sorted = axis[ind]
    x_resorted = []
    row_list = []
    for idx, elem in enumerate(x_sorted):
        if idx < len(x_sorted) - 1:
            row_list.append(elem)
            next_elem = x_sorted[idx + 1]
            if not math.isclose(elem, next_elem, rel_tol=1e-3, abs_tol=1e-3):
                x_resorted.append(row_list[0])
                row_list = []
        else:
            print("end reached!")
            row_list.append(elem)
            x_resorted.append(row_list[0])
    return x_resorted


class ODE:

    def __init__(self, mesh, ft, inlet_marker, wall_marker, mesh_t):
        self.time_discr = 33 # TODO: time discr. step size
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

        self.x0 = [[0.0 + i / 10.1, 0.0 + i / 10.1] for i in range(self.K)]
        self.lam_2_0 = [[0.0, 0.0] for i in range(self.K)]

    def ode_solving_step(self, u_dolf):

        def u_scp(x, t):
            point = np.array([x[0], x[1], 0])
            u_scp = self.dolfinx_to_numpy(point, u_dolf)
            return u_scp

        self.sol = []
        for k in range(self.K):
            x_sol_np = odeint(u_scp, self.x0[k], np.linspace(self.t0, self.T, self.time_discr))
            #x_sol_dolfinx = self.numpy_to_dolfin(x_sol_np)  # TODO: to dolfinx not necessary?
            self.sol.append(x_sol_np)
        return self.sol

    def adjoint_ode_solving_step(self, u_dolf, u_dk, x_k):
        self.adj_sol_dolfinx = []
        for k in range(self.K):
            grad_u_dolfin = ufl.grad(u_dolf(list(x_k[k])))  # TODO: to np
            U_dash = dolfinx.fem.functionspace(self.mesh,
                                               ("Lagrange", 2, (self.mesh.geometry.dim, self.mesh.geometry.dim)))
            grad_u = Function(U_dash)
            grad_u_expr = dolfinx.fem.Expression(grad_u_dolfin, U_dash.element.interpolation_points())
            grad_u.interpolate(grad_u_expr)
            # grad_u_np = []
            # for i, elem in enumerate(x_k[k]):
            #     point = np.concatenate((x_k[k][i], [0.0]))
            #     grad_u_np.append(self.dolfinx_to_numpy(point, grad_u))
            from IPython import embed
            embed()
            u_scp = lambda t, lam_2: grad_u(x_k[k](t)) @ (u_dolf(x_k[k](t)) - u_dk[k] + lam_2)
            adj_sol_np = odeint(u_scp, self.lam_2_0[k], np.linspace(-self.T, -self.t0), 33)
            adj_sol_np = adj_sol_np[::-1]
            adj_sol_dolfinx = self.numpy_to_dolfin(adj_sol_np, self.mesh, 3)  # TODO: to dolfinx
            self.sol.append(adj_sol_dolfinx)
        return self.adj_sol_dolfinx

    def dolfinx_to_numpy(self, point, func):
        func_dim = func.ufl_shape
        bb_tree = dolfinx.geometry.bb_tree(self.mesh, self.mesh.topology.dim)
        cells = []
        points_on_proc = []
        # Find cells whose bounding-box collide with the points
        cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, point)
        # Choose one of the cells that contains the point
        colliding_cells = dolfinx.geometry.compute_colliding_cells(self.mesh, cell_candidates, point)
        if len(colliding_cells.links(0)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(0)[0])
        points_on_proc = np.array(points_on_proc, dtype=np.float64)
        scp = func.eval(points_on_proc, cells)
        if len(scp) is 0:
            scp = np.ravel(np.zeros(func_dim))
        return scp

    def numpy_to_dolfin(self, x_np, mesh, dim):
        x_dolf = Function(self.X)
        if dim is 3:
            x, y, z = (mesh.geometry.x[:, 0], mesh.geometry.x[:, 1], self.mesh.geometry.x[:, 2])
            f = sc.interpolate.LinearNDInterpolator(list(zip(x, y)), x_np)

            def g(x):
                values = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
                temp = []
                for y_val in x[1]:
                    if y_val < 0.0:
                        temp.append(f(0.0))
                    elif y_val > 1.0:
                        temp.append(f(1.0))
                    else:
                        temp.append(f(y_val))
                values[1] = temp
                return values

            x_dolf.interpolate(g)
        else:
            x, y = (mesh.geometry.x[:, 0], mesh.geometry.x[:, 1])
            f = sc.interpolate.interp1d((x, y), x_np)

            def g(x):
                values = np.zeros((1, x.shape[1]), dtype=PETSc.ScalarType)
                temp = []
                for y_val in x[0]:
                    if y_val < self.t0:
                        temp.append(f(0.0))
                    elif y_val > self.T:
                        temp.append(f(1.0))
                    else:
                        temp.append(f(y_val))
                values[0] = temp
                return values

            x_dolf.interpolate(g)

        # TODO: loop through x,y,z to fit np.meshgrid
        # x_unique = np.unique(x)
        # xx = reconstruct_mesh_points(x)
        # yy = reconstruct_mesh_points(y)
        # zz = reconstruct_mesh_points(z)
        #f = sc.interpolate.RegularGridInterpolator((np.array(xx), np.array(yy)), u[:,0])


        return x_dolf