import json
import gmsh
import dolfinx
import mesh_init
import ufl
from ufl import (FacetNormal, Identity, Measure, TestFunction, TrialFunction,
                 as_vector, div, dot, inner, lhs, grad, nabla_grad, rhs, sym, system, SpatialCoordinate, inv,
                 sqrt, transpose, tr, ds, TestFunctions, TrialFunctions)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_vector, create_matrix, set_bc)
from dolfinx import log
from dolfinx.fem import (Constant, Function, functionspace,
                         assemble_scalar, dirichletbc, form, locate_dofs_topological, set_bc)
from petsc4py import PETSc
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
import numpy as np
import scipy as sc
from basix.ufl import element, mixed_element
import scifem


def petsc_matrix_to_numpy(mat):
    dim1, dim2 = mat.size
    return np.array([[mat.getValue(i, j) for i in range(dim1)] for j in range(dim2)])


def petsc_vector_to_numpy(vec):
    dim = vec.size
    return np.array([vec.getValue(i) for i in range(dim)])


class NavierStokes:

    def __init__(self, mesh, ft, inlet_marker, wall_marker, outlet_marker, experiment_number, np_path):
        self.bcu = None
        self.np_path = np_path
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
        self.fdim = mesh.topology.dim - 1
        with open("parameters.json", "r") as file:
            parameters = json.load(file)
            self.t0 = parameters["t0"]
            self.T = parameters["T"]
            self.h = parameters["dt"]
            self.viscosity = parameters["viscosity"]
            self.K = parameters["buoy count"]
            self.alpha = parameters["alpha"]
            self.mesh_boundary_x = parameters["mesh_boundary_x"]
            self.mesh_boundary_y = parameters["mesh_boundary_y"]
            self.delta = parameters["delta"]

        self.set_function_spaces()
        self.set_functions()
        self.set_boundary_conditions()

    def set_function_spaces(self):
        U_el = element("Lagrange", self.mesh.basix_cell(), 2, shape=(self.mesh.geometry.dim,))
        P_el = element("Lagrange", self.mesh.basix_cell(), 1)
        W_el = mixed_element([U_el, P_el])
        self.W = functionspace(self.mesh, W_el)
        self.U, _ = self.W.sub(0).collapse()

    def set_boundary_conditions(self):
        self.dx = ufl.Measure('dx', domain=self.mesh)
        self.ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=self.ft)

        def nonslip(x):
            values = np.zeros((2, x.shape[1]))
            return values

        u_nonslip = Function(self.U)
        u_nonslip.interpolate(nonslip)

        wall_dofs = locate_dofs_topological((self.W.sub(0), self.U), self.fdim, self.ft.find(self.wall_marker))
        bcu_wall = dirichletbc(u_nonslip, wall_dofs, self.W.sub(0))
        self.bcu = [bcu_wall]

    def set_functions(self):
        self.w = Function(self.W)
        self.u, self.p = ufl.split(self.w)
        # self.w.sub(0).x.array[:] = 0.0
        self.v, self.pr = ufl.split(TestFunction(self.W))

        self.w_adj = Function(self.W)
        self.u_adj, self.p_adj = TrialFunctions(self.W)
        self.v_adj, self.pr_adj = TestFunctions(self.W)

        self.u_r, self.p_r = TrialFunctions(self.W)
        self.v_r, self.pr_r = TestFunctions(self.W)

        self.u_r_adj, self.p_r_adj = TrialFunctions(self.W)
        self.v_r_adj, self.pr_r_adj = TestFunctions(self.W)

    def set_state_equations(self, q, u_r):
        self.set_functions()
        f = Constant(self.mesh, PETSc.ScalarType((0, 0)))
        a = self.viscosity * inner(grad(self.u), grad(self.v)) * self.dx # nabla u
        b = inner(self.p, div(self.v)) * self.dx # div(test)*p
        div_ = inner(self.pr, div(self.u)) * self.dx # div(u)*test_p
        c = inner(dot(self.u, nabla_grad(self.u)), self.v) * self.dx #nonlinear term
        u_dot_n = dot(self.u, FacetNormal(self.mesh)) # u * n
        psi_delta = 0.5 * (u_dot_n * ufl.tanh(u_dot_n / self.delta) - u_dot_n + self.delta) # psi(u*n)
        if u_r is not None:
            extra_bt = 0.5 * inner(psi_delta * self.u, self.v) * self.ds(self.inlet_marker) # 0.5 int((psi(u*n)*u - u_r) * test_u)_Gamma_1
        else:
            extra_bt = 0.5 * inner(psi_delta * self.u, self.v) * self.ds(self.inlet_marker) # 0.5 int((psi(u*n)*u - u_r) * test_u)_Gamma_1
        f_ = inner(f, self.v) * self.dx + inner(q, self.v) * self.ds(self.inlet_marker)  # control
        F = a + c + div_ - b + extra_bt - f_
        return F

    def state_solving_step(self, q, u_r, opt_step):
        print("solving primal NS")
        F = self.set_state_equations(q, u_r)
        problem = NonlinearProblem(F, self.w, bcs=self.bcu)
        solver = NewtonSolver(MPI.COMM_WORLD, problem)
        solver.convergence_criterion = "incremental"
        solver.rtol = 1e-6
        solver.report = True
        solver.max_it = 100
        ksp = solver.krylov_solver
        opts = PETSc.Options()
        option_prefix = ksp.getOptionsPrefix()
        opts[f"{option_prefix}ksp_type"] = "gmres"
        # opts[f"{option_prefix}ksp_rtol"] = 1.0e-8
        # opts[f"{option_prefix}pc_type"] = "hypre"
        # opts[f"{option_prefix}pc_hypre_type"] = "boomeramg"
        # opts[f"{option_prefix}pc_hypre_boomeramg_max_iter"] = 1
        # opts[f"{option_prefix}pc_hypre_boomeramg_cycle_type"] = "v"
        ksp.setFromOptions()
        #log.set_log_level(log.LogLevel.INFO)
        n, converged = solver.solve(self.w)
        assert (converged)
        print(f"Number of interations: {n:d}")

        # vtx_u = dolfinx.io.VTXWriter(self.mesh.comm, self.np_path + f"{self.experiment_number}_{opt_step}_u.bp", self.w.sub(0).collapse(),
        #                              engine="BP4")
        # vtx_u.write(0.0)
        # vtx_u.close()

        sol = Function(self.W)
        sol.x.array[:] = self.w.x.array
        return sol

    def set_adjoint_equation(self, u, lam_2, x, h, u_d, q, u_r):
        self.set_functions()
        a = self.viscosity * inner(grad(self.u_adj), grad(self.v_adj)) * self.dx
        c = (inner(dot(u, nabla_grad(self.v_adj)), self.u_adj) + inner(dot(self.v_adj, nabla_grad(u)),
                                                                       self.u_adj)) * self.dx
        b_form = inner(self.pr_adj, div(self.u_adj)) * self.dx
        div_ = inner(self.p_adj, div(self.v_adj)) * self.dx

        v_dot_n = dot(self.v_adj, FacetNormal(self.mesh))
        u_dot_n = dot(u, FacetNormal(self.mesh))

        cosh_quadr=dot(ufl.cosh(u_dot_n / self.delta),ufl.cosh(u_dot_n / self.delta))
        psi_d = 0.5 * (ufl.tanh(u_dot_n / self.delta) + u_dot_n / (self.delta* cosh_quadr) - 1)

        psi_delta = 0.5 * (u_dot_n * ufl.tanh(u_dot_n / self.delta) - u_dot_n + self.delta)
        d_delta = inner(psi_delta * self.v_adj, self.u_adj) * self.ds(self.inlet_marker)

        adj_extra_bt = 0.5 * (inner(v_dot_n * psi_d * (u), self.u_adj) * self.ds(self.inlet_marker) + d_delta)
        lhs_ = a + c + div_ - b_form + adj_extra_bt
        b = Function(self.W)
        b.x.array[:] = 0

        bb_tree = dolfinx.geometry.bb_tree(self.mesh, self.mesh.topology.dim)
        zeros = np.zeros((*x.shape[:2], 1))
        new_points = np.concatenate((x, zeros), axis=-1)

        u_values_arr = []
        for k, buoy_points in enumerate(new_points):
            cells = []
            points_on_proc = []
            cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, buoy_points)
            colliding_cells = dolfinx.geometry.compute_colliding_cells(self.mesh, cell_candidates,
                                                                       buoy_points)
            for i, point in enumerate(buoy_points):
                if len(colliding_cells.links(i)) > 0:
                    points_on_proc.append(point)
                    cells.append(colliding_cells.links(i)[0])

            u_values = u.eval(points_on_proc, cells)
            gamma = (u_d[k, : , :] - u_values + lam_2[k,:,:])
            ps1 = scifem.PointSource(self.W.sub(0).sub(0), new_points, magnitude=h * gamma[:, 0])
            ps2 = scifem.PointSource(self.W.sub(0).sub(1), new_points, magnitude=h * gamma[:, 1])
            ps1.apply_to_vector(b)
            ps2.apply_to_vector(b)
            u_values_arr.append(u_values)

        # point_shape = new_points.shape
        # new_points = new_points.reshape(point_shape[0] * point_shape[1], point_shape[2])
        #
        # cells = []
        # points_on_proc = []
        # cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, new_points)
        # colliding_cells = dolfinx.geometry.compute_colliding_cells(self.mesh, cell_candidates,
        #                                                            new_points)
        # for i, point in enumerate(new_points):
        #     if len(colliding_cells.links(i)) > 0:
        #         points_on_proc.append(point)
        #         cells.append(colliding_cells.links(i)[0])
        #
        # u_values = u.eval(points_on_proc, cells)
        # ud = u_d.reshape(u_d.shape[0] * u_d.shape[1], u_d.shape[2])
        # lam2 = lam_2.reshape(lam_2.shape[0] * lam_2.shape[1], lam_2.shape[2])
        # gamma = (ud - u_values + lam2)
        # ps1 = scifem.PointSource(self.W.sub(0).sub(0), new_points, magnitude=h * gamma[:, 0])
        # ps2 = scifem.PointSource(self.W.sub(0).sub(1), new_points, magnitude=h * gamma[:, 1])
        # ps1.apply_to_vector(b)
        # ps2.apply_to_vector(b)

        lhs_form = form(lhs_)
        apply_lifting(b.x.petsc_vec, [lhs_form], [self.bcu])
        b.x.scatter_reverse(dolfinx.la.InsertMode.add)
        dolfinx.fem.petsc.set_bc(b.x.petsc_vec, self.bcu)
        b.x.scatter_forward()
        A = dolfinx.fem.petsc.assemble_matrix(lhs_form, bcs=self.bcu)
        A.assemble()

        qnorm = form(dot(q, q) * self.ds(self.inlet_marker))
        comm = q.function_space.mesh.comm
        E = comm.allreduce(assemble_scalar(qnorm), MPI.SUM)
        J = 0.5 * np.sum(np.sum(h * (np.linalg.norm(np.array(u_values_arr) - u_d, axis=2) ** 2), axis=1)) + (self.alpha / 2) * E
        return A, b, J, u_values_arr

    def adjoint_state_solving_step(self, u, lam_2, x, h, u_d, q, u_r):
        print("solving adjoint NS")
        A, rhs, J, u_values = self.set_adjoint_equation(u, lam_2, x, h, u_d, q, u_r)
        ksp = PETSc.KSP().create(self.mesh.comm)
        ksp.setOperators(A)
        ksp.setType(PETSc.KSP.Type.PREONLY)
        ksp.getPC().setType(PETSc.PC.Type.LU)
        ksp.getPC().setFactorSolverType("mumps")
        up = dolfinx.fem.Function(self.W)
        ksp.solve(rhs.x.petsc_vec, up.x.petsc_vec)
        up.x.scatter_forward()

        return up, J, u_values

    def set_stokes_state(self, q):
        self.set_functions()
        f = Constant(self.mesh, PETSc.ScalarType((0, 0)))
        a = self.viscosity * inner(grad(self.u_r), grad(self.v_r)) * self.dx
        b = inner(self.p_r, div(self.v_r)) * self.dx
        div_ = inner(self.pr_r, div(self.u_r)) * self.dx
        f_ = inner(f, self.v_r) * self.dx + inner(q, self.v_r) * self.ds(self.inlet_marker)
        F = a + div_ - b  # - extra_bt
        a = form(F)
        L1 = form(f_)
        A = assemble_matrix(a, bcs=self.bcu)
        A.assemble()
        b1 = create_vector(L1)
        return A, b1, a, L1

    def set_adjoint_stokes(self, u):
        self.set_functions()
        f = Constant(self.mesh, PETSc.ScalarType((0, 0)))
        a = self.viscosity * inner(grad(self.u_r_adj), grad(self.v_r_adj)) * self.dx
        b = inner(self.p_r_adj, div(self.v_r_adj)) * self.dx
        div_ = inner(self.pr_r_adj, div(self.u_r_adj)) * self.dx
        f_ = inner(u, self.v_r_adj) * self.dx
        F = a + div_ - b  # - extra_bt
        a = form(F)
        L1 = form(f_)
        A = assemble_matrix(a, bcs=self.bcu)
        A.assemble()
        b1 = create_vector(L1)
        return A, b1, a, L1

    def solve_stokes_step(self, q):
        A, b, a, L = self.set_stokes_state(q)
        w_s = Function(self.W)
        solver1 = PETSc.KSP().create(self.mesh.comm)
        solver1.setOperators(A)
        solver1.setType(PETSc.KSP.Type.BCGS)
        pc1 = solver1.getPC()
        pc1.setType(PETSc.PC.Type.JACOBI)

        A.zeroEntries()
        assemble_matrix(A, a, bcs=self.bcu)
        A.assemble()
        with b.localForm() as loc:
            loc.set(0)
        assemble_vector(b, L)
        apply_lifting(b, [a], [self.bcu])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, self.bcu)
        solver1.solve(b, w_s.x.petsc_vec)
        w_s.x.scatter_forward()
        return w_s

    def solve_adoint_stokes_step(self, u):
        A, b, a, L = self.set_adjoint_stokes(u)
        w_s = Function(self.W)
        solver1 = PETSc.KSP().create(self.mesh.comm)
        solver1.setOperators(A)
        solver1.setType(PETSc.KSP.Type.BCGS)
        pc1 = solver1.getPC()
        pc1.setType(PETSc.PC.Type.JACOBI)

        A.zeroEntries()
        assemble_matrix(A, a, bcs=self.bcu)
        A.assemble()
        with b.localForm() as loc:
            loc.set(0)
        assemble_vector(b, L)
        apply_lifting(b, [a], [self.bcu])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, self.bcu)
        solver1.solve(b, w_s.x.petsc_vec)
        w_s.x.scatter_forward()
        return w_s

    def old_code(self):
        a = 1
        # for b in range(self.K):
        #     points = np.array([x[b, :, :][:,0], x[b, :, :][:, 1], np.zeros(len(x[b, :, :][:,0]))]).T
        #     point_marker = 5
        #     num_entities_local = self.mesh.topology.index_map(self.mesh.topology.dim).size_local + self.mesh.topology.index_map(self.mesh.topology.dim).num_ghosts
        #     entities = np.arange(num_entities_local, dtype=np.int32)
        #     midpoint_tree = dolfinx.geometry.create_midpoint_tree(self.mesh, self.mesh.topology.dim, entities)
        #     #facet_indices = dolfinx.geometry.compute_closest_entity(bb_tree,midpoint_tree,self.mesh, points)
        #     facet_indices = []
        #
        #     cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, points)
        #     facet_indices = np.unique(dolfinx.geometry.compute_colliding_cells(self.mesh, cell_candidates, points).array)
        #
        #     facet_tags = dolfinx.mesh.meshtags(self.mesh, self.mesh.topology.dim-1, facet_indices, np.full_like(facet_indices, point_marker))
        #
        #
        #     ds = ufl.Measure("dS",domain=self.mesh, subdomain_data=facet_tags, subdomain_id=point_marker)
        #
        #     # Term associated with points
        #     for l2 in lam_2[b, :, :]:
        #         F += l2*self.v_adj * ds
        #     #line_integral_1 += lam_2[b, :, :] * self.v_adj * ds

        # a_form = form(a)
        # K = create_matrix(a_form)
        # K.zeroEntries()
        # assemble_matrix(K, a_form, bcs=self.bcu)
        # K.assemble()
        #
        # b_form = form(b)
        # Bt = create_matrix(b_form)
        # Bt.zeroEntries()
        # assemble_matrix(Bt, b_form, bcs=self.bcus)
        # Bt.assemble()
        #
        # M_form = form(m)
        # M = create_matrix(M_form)
        # M.zeroEntries()
        # assemble_matrix(M, M_form, bcs=[self.bcp])
        # M.assemble()
        #
        # c_form = form(c)
        # C = create_matrix(c_form)
        # C.zeroEntries()
        # assemble_matrix(C, c_form, bcs=self.bcu)
        # C.assemble()
        #
        # f_form = form(f_)
        # f_vector = assemble_vector(f_form)
        # f_vector.assemble()
        # set_bc(f_vector, bcs=self.bcus)
        # K, Bt, C, f, M, a_form, b_form = self.set_state_equations(q)
        # K_np = petsc_matrix_to_numpy(K)
        # K_np_inv = np.linalg.inv(K_np)
        # K_inv = PETSc.Mat().createDense(K_np_inv.shape, array=K_np_inv)

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
        # B = PETSc.Mat().createTranspose(Bt)
        # solver1 = PETSc.KSP().create(MPI.COMM_WORLD)
        # solver1.setOperators(K + self.alpha * C)
        # solver1.setType(PETSc.KSP.Type.BCGS)
        # pc1 = solver1.getPC()   # preconditioner
        # pc1.setType(PETSc.PC.Type.JACOBI)
        #
        # solver2 = PETSc.KSP().create(MPI.COMM_WORLD)
        # solver2.setOperators(B*K_inv*Bt)
        # solver2.setType(PETSc.KSP.Type.CG)
        # pc2 = solver2.getPC()
        # pc2.setType(PETSc.PC.Type.SOR)
        #
        # solver3 = PETSc.KSP().create(MPI.COMM_WORLD)
        # solver3.setOperators(K)
        # solver3.setType(PETSc.KSP.Type.CG)
        # pc3 = solver3.getPC()
        # pc3.setType(PETSc.PC.Type.SOR)
        #
        # u_hat2 = Function(self.U)
        #
        # L1 = form(f + (self.alpha - 1) * inner(dot(self.u_k, nabla_grad(self.u_k)), self.v) * self.dx + inner(div(self.v), self.p_k) * self.dx)
        # b1 = create_vector(L1)
        #
        # L2 = form(-inner(div(self.u_hat), self.pr) * self.dx)
        # b2 = create_vector(L2)
        #
        # L3 = form(self.viscosity * inner(grad(self.u_hat) + grad(self.u_hat).T, grad(self.v)) * self.dx + inner(div(self.v), self.p_hat) * self.dx)
        # b3 = create_vector(L3)
        #
        # from IPython import embed
        # embed()
        # for i in range(100):
        #     assemble_vector(b1, L1)
        #     apply_lifting(b1, [a_form], [self.bcus])
        #     b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        #     set_bc(b1, self.bcus)
        #     solver1.solve(b1, self.u_hat.x.petsc_vec)
        #     self.u_hat.x.scatter_forward()
        #
        #     # u_.x.petsc_vec.axpy(1, self.u_hat.x.petsc_vec)
        #     # u_.x.scatter_forward()
        #
        #     assemble_vector(b2, L2)
        #     apply_lifting(b2, [b_form], [self.bcus])
        #     b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        #     set_bc(b2, self.bcus)
        #     solver2.solve(b2, self.p_hat.x.petsc_vec)
        #     self.p_hat.x.scatter_forward()
        #
        #     # p_.x.petsc_vec.axpy(1, self.p_hat.x.petsc_vec)
        #     # p_.x.scatter_forward()
        #
        #     assemble_vector(b3, L3)
        #     apply_lifting(b3, [a_form], [self.bcus])
        #     b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        #     set_bc(b3, self.bcus)
        #     solver3.solve(b3, u_hat2.x.petsc_vec)
        #     u_hat2.x.scatter_forward()
        #
        #     self.p_s.x.array[:] = self.p_hat.x.array + self.p_k.x.array
        #     self.u_s.x.array[:] = self.gamma * u_hat2.x.array + (1-self.gamma) * self.u_k.x.array
        #
        #     u_err = np.linalg.norm(self.u_s.x.array - self.u_k.x.array)
        #     p_err = np.linalg.norm(self.p_s.x.array - self.p_k.x.array)
        #     print(u_err)
        #     print(p_err)
        #
        #     self.u_k.x.array[:] = self.u_s.x.array
        #     self.p_k.x.array[:] = self.p_s.x.array
        #     self.u_k.x.scatter_forward()
        #     self.p_k.x.scatter_forward()
        #     # from IPython import embed
        #     # embed()
        #
        #     print(i)
        return a
