import json
import gmsh
import dolfinx
import mesh_init
import ufl
from ufl import (FacetNormal, Identity, Measure, TestFunction, TrialFunction,
                 as_vector, div, dot, inner, lhs, grad, nabla_grad, rhs, sym, system, SpatialCoordinate, inv,
                 sqrt, transpose, tr, ds)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_vector, create_matrix, set_bc)
from dolfinx.fem import (Constant, Function, functionspace,
                         assemble_scalar, dirichletbc, form, locate_dofs_topological, set_bc, form)
from petsc4py import PETSc
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
import numpy as np
import scipy as sc
from basix.ufl import element, mixed_element


def petsc_matrix_to_numpy(mat):
    dim1, dim2 = mat.size
    return np.array([[mat.getValue(i, j) for i in range(dim1)] for j in range(dim2)])


def petsc_vector_to_numpy(vec):
    dim = vec.size
    return np.array([vec.getValue(i) for i in range(dim)])


class NavierStokes:

    def __init__(self, mesh, ft, inlet_marker, wall_marker, outlet_marker, obstacle_marker, experiment_number):
        self.np_path = "results/experiments/"
        self.experiment_number = experiment_number
        self.p = None
        self.P = None
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
        self.outlet_marker = outlet_marker
        self.obstacle_marker = obstacle_marker
        self.fdim = mesh.topology.dim - 1
        with open("parameters.json", "r") as file:
            parameters = json.load(file)
            self.t0 = parameters["t0"]
            self.T = parameters["T"]
            self.h = parameters["dt"]
            self.viscosity = parameters["viscosity"]
            self.K = parameters["buoy count"]
            self.alpha = parameters["alpha"]
            self.gamma = parameters["gamma"]

        self.set_function_spaces()
        self.set_functions()
        self.set_boundary_conditions()

    def set_function_spaces(self):
        self.Q = dolfinx.fem.functionspace(self.mesh, ("Lagrange", 2, (self.mesh.geometry.dim,)))

        self.U = dolfinx.fem.functionspace(self.mesh, ("Lagrange", 2, (self.mesh.geometry.dim,)))
        self.P = dolfinx.fem.functionspace(self.mesh, ("Lagrange", 1))  # , (self.mesh.geometry.dim,)))
        # up_element = mixed_element([self.U.ufl_element(), self.P.ufl_element()])
        # self.W = dolfinx.fem.functionspace(self.mesh, up_element)

        self.U_adj = self.U.clone()
        self.Q_adj = self.Q.clone()

    def set_boundary_conditions(self):
        # n = FacetNormal(self.mesh)
        u_nonslip = np.array((0,) * self.mesh.geometry.dim, dtype=PETSc.ScalarType)
        bcu_wall = dirichletbc(u_nonslip, locate_dofs_topological(self.U, self.fdim, self.ft.find(self.wall_marker)),
                              self.U)
        self.dx = ufl.Measure('dx', domain=self.mesh)
        self.ds = ufl.Measure("ds", subdomain_id=self.outlet_marker)

        class InletVelocity():
            def __init__(self, t):
                self.t = t

            def __call__(self, x):
                values = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
                values[0] = 2 * 1.5 * x[1] * (0.41 - x[1]) / (0.41 ** 2)
                return values

        u_inlet = Function(self.U)
        inlet_velocity = InletVelocity(0)
        u_inlet.interpolate(inlet_velocity)
        bcu_inlet = dirichletbc(u_inlet, locate_dofs_topological(self.U, self.mesh.topology.dim - 1,
                                                                     self.ft.find(self.inlet_marker))) #TODO: remove inlet to let control work

        bcp_outlet = dirichletbc(PETSc.ScalarType(0), locate_dofs_topological(self.P, self.fdim, self.ft.find(self.outlet_marker)), self.P)
        bcu_obstacle = dirichletbc(u_nonslip, locate_dofs_topological(self.U, self.fdim, self.ft.find(self.obstacle_marker)), self.U)
        self.bcu = [bcu_inlet, bcu_obstacle, bcu_wall]
        self.bcp = bcp_outlet

    def set_functions(self):
        (self.u, self.p) = (ufl.TrialFunction(self.U), ufl.TrialFunction(self.P))
        (self.v, self.pr) = (ufl.TestFunction(self.U), ufl.TestFunction(self.P))
        self.u_k, self.p_k = (Function(self.U), Function(self.P))
        self.u_s = Function(self.U)
        self.p_s = Function(self.P)
        self.u_hat = Function(self.U)
        self.p_hat = Function(self.P)

        # self.q = Function(self.Q)
        (self.lam_1, self.delta_u) = (TrialFunction(self.U_adj), TestFunction(self.U_adj))

        # self.x = [Function(self.X) for _ in range(self.K)]
        # self.lam_2 = [Function(self.X_adj) for _ in range(self.K)]
        #
        # self.u_d_dolfin = [Constant(self.mesh, PETSc.ScalarType((0, 0))) for _ in range(self.K)]

    def set_state_equations(self, q):
        f = Constant(self.mesh, PETSc.ScalarType((0, 0)))
        a = self.viscosity * inner(grad(self.u) + grad(self.u).T, grad(self.v)) * self.dx
        b = inner(self.p, div(self.v)) * self.dx
        m = inner(self.p, self.pr) * self.dx
        c = inner(dot(self.u_k, nabla_grad(self.u)), self.v) * self.dx

        self.bcus = self.bcu
        self.bcus.append(self.bcp)
        a_form = form(a)
        K = create_matrix(a_form)
        K.zeroEntries()
        assemble_matrix(K, a_form, bcs=self.bcu)
        K.assemble()

        b_form = form(b)
        Bt = create_matrix(b_form)
        Bt.zeroEntries()
        assemble_matrix(Bt, b_form, bcs=self.bcus)
        Bt.assemble()

        M_form = form(m)
        M = create_matrix(M_form)
        M.zeroEntries()
        assemble_matrix(M, M_form, bcs=[self.bcp])
        M.assemble()

        c_form = form(c)
        C = create_matrix(c_form)
        C.zeroEntries()
        assemble_matrix(C, c_form, bcs=self.bcu)
        C.assemble()

        f_ = inner(f, self.v) * self.dx #+ inner(q, self.v) * self.ds TODO: control here
        f_form = form(f_)
        f_vector = assemble_vector(f_form)
        f_vector.assemble()
        set_bc(f_vector, bcs=self.bcus)

        return K, Bt, C, f_, M, a_form, b_form

    def state_solving_step(self, q):
        K, Bt, C, f, M, a_form, b_form = self.set_state_equations(q)
        K_np = petsc_matrix_to_numpy(K)
        K_np_inv = np.linalg.inv(K_np)
        K_inv = PETSc.Mat().createDense(K_np_inv.shape, array=K_np_inv)

        # K_np = petsc_matrix_to_numpy(K)
        # f_np = petsc_vector_to_numpy(f)
        # Bt_np = petsc_matrix_to_numpy(Bt).T
        # M_np = petsc_matrix_to_numpy(M).T
        # M_inv = np.linalg.inv(M_np)
        # lhs2 = M_inv @ (Bt_np.T @ np.linalg.inv(K_np) @ Bt_np)
        # i = 0
        # while True:
        #     # for i in range(100):
        #
        #     i += 1
        #     c = inner(dot(self.u_k, nabla_grad(self.u)), self.v) * self.dx
        #     c_form = form(c)
        #     C = assemble_matrix(c_form, bcs=self.bcus)
        #     C.assemble()
        #     C_np = petsc_matrix_to_numpy(C)
        #     lhs1 = K_np + self.alpha * C_np
        #     rhs1 = f_np + (self.alpha - 1) * C_np @ self.u_k.x.array - Bt_np @ self.p_k.x.array
        #
        #     u_hat_s = sc.linalg.solve(lhs1, rhs1)
        #
        #     print(K_np[m,n])
        #     intr = self.u_k.x.array
        #     from IPython import embed
        #     embed()
        #     delta_p = sc.linalg.solve(lhs2, M_inv @ (Bt_np.T @ u_hat_s))
        #
        #     u_hat = sc.linalg.solve(K_np, K_np @ u_hat_s - Bt_np @ delta_p)
        #
        #     self.p_s.x.array[:] = delta_p + self.p_k.x.array
        #
        #     self.u_s.x.array[:] = self.gamma * u_hat + (1 - self.gamma) * self.u_k.x.array
        #
        #     # p_old = self.p_s.x.array[:]
        #     # u_old = self.u_s.x.array[:]
        #
        #     u_err = np.linalg.norm(self.u_s.x.array - self.u_k.x.array)
        #     p_err = np.linalg.norm(self.p_s.x.array - self.p_k.x.array)
        #     print(i)
        #     print(u_err)
        #     print(p_err)
        #     if u_err < 1e-6 and p_err < 1e-6:
        #         break
        #
        #     self.p_k.x.array[:] = self.p_s.x.array
        #     self.u_k.x.array[:] = self.u_s.x.array
        #
        # self.save_vector_fields()
        B = PETSc.Mat().createTranspose(Bt)
        solver1 = PETSc.KSP().create(MPI.COMM_WORLD)
        solver1.setOperators(K + self.alpha * C)
        solver1.setType(PETSc.KSP.Type.BCGS)
        pc1 = solver1.getPC()   # preconditioner
        pc1.setType(PETSc.PC.Type.JACOBI)

        solver2 = PETSc.KSP().create(MPI.COMM_WORLD)
        solver2.setOperators(B*K_inv*Bt)
        solver2.setType(PETSc.KSP.Type.CG)
        pc2 = solver2.getPC()
        pc2.setType(PETSc.PC.Type.SOR)

        solver3 = PETSc.KSP().create(MPI.COMM_WORLD)
        solver3.setOperators(K)
        solver3.setType(PETSc.KSP.Type.CG)
        pc3 = solver3.getPC()
        pc3.setType(PETSc.PC.Type.SOR)

        u_hat2 = Function(self.U)

        L1 = form(f + (self.alpha - 1) * inner(dot(self.u_k, nabla_grad(self.u_k)), self.v) * self.dx + inner(div(self.v), self.p_k) * self.dx)
        b1 = create_vector(L1)

        L2 = form(-inner(div(self.u_hat), self.pr) * self.dx)
        b2 = create_vector(L2)

        L3 = form(self.viscosity * inner(grad(self.u_hat) + grad(self.u_hat).T, grad(self.v)) * self.dx + inner(div(self.v), self.p_hat) * self.dx)
        b3 = create_vector(L3)

        from IPython import embed
        embed()
        for i in range(100):
            assemble_vector(b1, L1)
            apply_lifting(b1, [a_form], [self.bcus])
            b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b1, self.bcus)
            solver1.solve(b1, self.u_hat.x.petsc_vec)
            self.u_hat.x.scatter_forward()

            # u_.x.petsc_vec.axpy(1, self.u_hat.x.petsc_vec)
            # u_.x.scatter_forward()

            assemble_vector(b2, L2)
            apply_lifting(b2, [b_form], [self.bcus])
            b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b2, self.bcus)
            solver2.solve(b2, self.p_hat.x.petsc_vec)
            self.p_hat.x.scatter_forward()

            # p_.x.petsc_vec.axpy(1, self.p_hat.x.petsc_vec)
            # p_.x.scatter_forward()

            assemble_vector(b3, L3)
            apply_lifting(b3, [a_form], [self.bcus])
            b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b3, self.bcus)
            solver3.solve(b3, u_hat2.x.petsc_vec)
            u_hat2.x.scatter_forward()

            self.p_s.x.array[:] = self.p_hat.x.array + self.p_k.x.array
            self.u_s.x.array[:] = self.gamma * u_hat2.x.array + (1-self.gamma) * self.u_k.x.array

            u_err = np.linalg.norm(self.u_s.x.array - self.u_k.x.array)
            p_err = np.linalg.norm(self.p_s.x.array - self.p_k.x.array)
            print(u_err)
            print(p_err)

            self.u_k.x.array[:] = self.u_s.x.array
            self.p_k.x.array[:] = self.p_s.x.array
            self.u_k.x.scatter_forward()
            self.p_k.x.scatter_forward()
            # from IPython import embed
            # embed()

            print(i)

        return self.u_s

    def set_adjoint_equation(self, lam_2):
        F2 = self.viscosity * inner(grad(self.lam_1), grad(self.delta_u)) * self.dx
        F2 += inner(self.lam_1, dot(self.delta_u, nabla_grad(self.u))) * self.dx
        F2 += inner(self.lam_1, dot(self.u, nabla_grad(self.delta_u))) * self.dx
        for i in range(0, self.K):
            F2 += inner(lam_2[i], self.delta_u(self.x[i])) * self.dt
        for i in range(0, self.K):
            F2 += inner(self.u - self.u_d_dolfin[i], self.delta_u(self.x[i])) * self.dt
        return F2

    def adjoint_state_solving_step(self, lam_2, u, x):
        F_adj = self.set_adjoint_equation(lam_2)
        adj_problem = dolfinx.fem.petsc.NonlinearProblem(F_adj, self.lam_1)
        adj_solver = NewtonSolver(MPI.COMM_WORLD, adj_problem)
        adj_solver.convergence_criterion = "incremental"
        adj_solver.report = True
        adj_solver.rtol = 1e-4
        adj_solver.max_it = 1000
        n, converged = adj_solver.solve(self.lam_1)
        return self.lam_1

    def save_vector_fields(self):
        with dolfinx.io.VTXWriter(MPI.COMM_WORLD, self.np_path + f"{self.experiment_number}_pressure.bp", [self.p_s],
                                  engine="BP4") as vtx:
            vtx.write(0.0)
        with dolfinx.io.VTXWriter(MPI.COMM_WORLD, self.np_path + f"{self.experiment_number}_u.bp", [self.u_s],
                                  engine="BP4") as vtx:
            vtx.write(0.0)
