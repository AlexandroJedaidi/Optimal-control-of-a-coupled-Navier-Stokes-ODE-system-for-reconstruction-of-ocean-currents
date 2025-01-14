import json
import sys

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
import random


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

    def __init__(self, mesh, ft, inlet_marker, wall_marker):
        self.time_interval = None
        self.sol = None
        self.X = None
        self.X_adj = None
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
        self.Q = None
        self.mesh = mesh
        self.ft = ft
        self.inlet_marker = inlet_marker
        self.wall_marker = wall_marker
        self.fdim = mesh.topology.dim - 1
        with open("parameters.json", "r") as file:
            parameters = json.load(file)
            self.t0 = parameters["t0"]
            self.T = parameters["T"]
            self.h = parameters["dt"]
            self.viscosity = parameters["viscosity"]
            self.K = parameters["buoy count"]
            self.mesh_boundary_x = parameters["mesh_boundary_x"]
            self.mesh_boundary_y = parameters["mesh_boundary_y"]

        self.debug_path = f"results/debug/"
        self.set_function_spaces()
        self.set_functions()

    def set_function_spaces(self):

        self.time_interval = np.linspace(self.t0, self.T, int(self.T / self.h))
        self.N = len(self.time_interval[1:])

    def set_functions(self):
        self.x = np.zeros((self.K, int(self.T / self.h), self.mesh.geometry.dim))
        # self.x[:, 0, 0] = np.array([random.uniform(self.mesh_boundary_x[0], self.mesh_boundary_x[1]) for _ in range(self.K)])
        # self.x[:, 0, 0] = np.array([random.uniform(self.mesh_boundary_x[0], self.mesh_boundary_x[1]) for _ in range(self.K)])
        self.x[:, 0, 0] = np.array([0.1 for _ in range(self.K)])
        self.x[:, 0, 1] = np.array([0.5 for _ in range(self.K)])


        self.lam_2 = np.zeros((self.K, int(self.T / self.h), self.mesh.geometry.dim))

        self.u_d = np.zeros((self.K, int(self.T / self.h), self.mesh.geometry.dim))

    def ode_solving_step(self, u):
        # explicit euler
        bb_tree = dolfinx.geometry.bb_tree(self.mesh, self.mesh.topology.dim)
        for b in range(self.K):
            for k, t_k in enumerate(self.time_interval[:-1]):
                print("time: " + str(t_k))
                point = np.array([self.x[b, k, :][0].item(), self.x[b, k, :][1].item(), 0])
                cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, point)
                colliding_cells = dolfinx.geometry.compute_colliding_cells(self.mesh, cell_candidates, point).array
                if len(colliding_cells) == 0:
                    print("no colliding cells at time:" + str(t_k))
                    print("buoy number: " + str(b))
                    print("original coordinate: " + str(self.x[b, 0, :][0].item()) + " " + str(self.x[b, 0, :][1].item()))
                    print("actual coordinate: " + str(point))
                    exit(0)
                u_values = u.eval(point, colliding_cells[0])
                self.x[b, k + 1, :] = self.x[b, k, :] + self.h * u_values
        return self.x

    def adjoint_ode_solving_step(self, u):
        grad_u = ufl.grad(u)
        U_grad_fp = dolfinx.fem.functionspace(self.mesh,
                                              ("Lagrange", 2, (self.mesh.geometry.dim, self.mesh.geometry.dim)))
        u_grad_expr = dolfinx.fem.Expression(grad_u, U_grad_fp.element.interpolation_points())
        u_grad_fct = dolfinx.fem.Function(U_grad_fp)
        u_grad_fct.interpolate(u_grad_expr)
        bb_tree = dolfinx.geometry.bb_tree(self.mesh, self.mesh.topology.dim)
        for b in range(self.K):
            N = len(self.time_interval[1:])
            for k in range(N-1, -1, -1):
                point = np.array([self.x[b, k, :][0].item(), self.x[b, k, :][1].item(), 0])
                cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, point)
                colliding_cells = dolfinx.geometry.compute_colliding_cells(self.mesh, cell_candidates, point).array
                if len(colliding_cells) < 0:
                    print("no colliding cells")
                    from IPython import embed
                    embed()
                grad_u_values = u_grad_fct.eval(point, colliding_cells[0])
                from IPython import embed; embed()
                grad_u_matr = np.array([[grad_u_values[0].item(), grad_u_values[1].item()],
                                        [grad_u_values[2].item(), grad_u_values[3].item()]])

                u_values = u.eval(point, colliding_cells[0])
                A = (np.identity(2) + self.h * grad_u_matr.T)
                b_vec = self.lam_2[b, k+1, :] - self.h * grad_u_matr.T @ (u_values - self.u_d[b, k, :])
                self.lam_2[b, k, :] = np.linalg.solve(A, b_vec)

                # self.lam_2[b, k - 1, :] = self.x[b, k, :] - self.h * (
                #         grad_u_matr.T @ (u_values - self.u_d[b, k, :] + self.lam_2[b, k, :]))
        return self.lam_2

    def old_code(self):
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
            # f = sc.interpolate.RegularGridInterpolator((np.array(xx), np.array(yy)), u[:,0])

            return x_dolf

        old_code = """for k in range(self.K):
                    grad_u_dolfin = ufl.grad(u_dolf(list(x_k[k])))  # TODO: to np
                    U_dash = dolfinx.fem.functionspace(self.mesh,
                                                       ("Lagrange", 2, (self.mesh.geometry.dim, self.mesh.geometry.dim)))
                    grad_u = Function(U_dash)
                    grad_u_expr = dolfinx.fem.Expression(grad_u_dolfin, U_dash.element.interpolation_points())
                    grad_u.interpolate(grad_u_expr)

                    u_scp = lambda t, lam_2: grad_u(x_k[k](t)) @ (u_dolf(x_k[k](t)) - u_dk[k] + lam_2)
                    adj_sol_np = odeint(u_scp, self.lam_2_0[k], np.linspace(-self.T, -self.t0, self.time_discr))
                    adj_sol_np = adj_sol_np[::-1]
                    adj_sol_dolfinx = self.numpy_to_dolfin(adj_sol_np, self.mesh, 3)  # TODO: to dolfinx
                    self.sol.append(adj_sol_dolfinx)"""
        old_code = """time_interval = self.mesh_t.geometry.x[:, 0]
                N = time_interval.shape[0]
                h = self.T / N
                x_dolf_list = []
                for b in range(self.K):
                    x_sol = [self.x0[b]]
                    for k, t_k in enumerate(time_interval):
                        x_i = x_sol[k] + h * u_dolf(self.x[b](t_k))
                        x_sol.append(x_i)
                    # from IPython import embed
                    # embed()
                    f = sc.interpolate.interp1d(time_interval, x_sol[:N].x.array)
                    x_dolf = Function(self.X)

                    def g(x):
                        values = np.zeros((2, self.x.shape[1]), dtype=PETSc.ScalarType)
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
                    x_dolf_list.append(x_dolf)
                def u_scp(x, t):
                    point = np.array([x[0], x[1], 0])
                    u_scp = self.dolfinx_to_numpy(point, u_dolf)
                    return u_scp

                for k in range(self.K):
                    x_sol_np = odeint(u_scp, self.x0[k], np.linspace(self.t0, self.T, self.time_discr))
                    #x_sol_dolfinx = self.numpy_to_dolfin(x_sol_np)  # TODO: to dolfinx not necessary?
                    from IPython import embed
                    embed()
                    x_sol_expr = dolfinx.fem.Expression(x_sol_np, self.X.element.interpolation_points())
                    self.sol.append(x_sol_np)"""
