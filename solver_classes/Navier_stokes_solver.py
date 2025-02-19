import json
from wsgiref.simple_server import WSGIServer

import gmsh
import dolfinx
import mesh_init
import ufl
from ufl import (FacetNormal, Identity, Measure, TestFunction, TrialFunction,
                 as_vector, div, dot, inner, lhs, grad, nabla_grad, rhs, sym, system, SpatialCoordinate, inv,
                 sqrt, transpose, tr, TestFunctions, TrialFunctions)
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


with open("parameters.json", "r") as file:
    parameters = json.load(file)
    t0 = parameters["t0"]
    T = parameters["T"]
    h = parameters["dt"]
    viscosity = parameters["viscosity"]
    K = parameters["buoy count"]
    alpha = parameters["alpha"]
    mesh_boundary_x = parameters["mesh_boundary_x"]
    mesh_boundary_y = parameters["mesh_boundary_y"]
    delta = parameters["delta"]


def set_state_equations(primal_NS_variable_package, q, u_r, mesh, dx, ds, inlet_marker):
    w, u, p, v, pr = primal_NS_variable_package
    f = Constant(mesh, PETSc.ScalarType((0, 0)))
    a = viscosity * inner(grad(u), grad(v)) * dx  # nabla u
    b = inner(p, div(v)) * dx  # div(test)*p
    div_ = inner(pr, div(u)) * dx  # div(u)*test_p
    c = inner(nabla_grad(u) * u, v) * dx  # nonlinear term
    u_dot_n = dot(u, FacetNormal(mesh))  # u * n
    psi_delta = 0.5 * (u_dot_n * ufl.tanh(u_dot_n / delta) - u_dot_n + delta)  # psi(u*n)
    if u_r is not None:
        extra_bt = 0.5 * inner(psi_delta * u, v) * ds(
            inlet_marker)  # 0.5 int((psi(u*n)*u - u_r) * test_u)_Gamma_1
    else:
        extra_bt = 0.5 * inner(psi_delta * u, v) * ds(
            inlet_marker)  # 0.5 int((psi(u*n)*u - u_r) * test_u)_Gamma_1
    f_ = inner(f, v) * dx + inner(q, v) * ds(inlet_marker)  # control
    F = a + c + div_ - b + extra_bt - f_
    return F


def state_solving_step(primal_NS_variable_package, q, u_r, opt_step, mesh, dx, ds, inlet_marker, bcu, W):
    print("solving primal NS")
    F = set_state_equations(primal_NS_variable_package, q, u_r, mesh, dx, ds, inlet_marker)
    w = primal_NS_variable_package[0]
    problem = NonlinearProblem(F, w, bcs=bcu)
    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-10
    solver.report = True
    solver.max_it = 500
    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "preonly"
    # opts[f"{option_prefix}ksp_rtol"] = 1.0e-8
    opts[f"{option_prefix}pc_type"] = "lu"
    # opts[f"{option_prefix}pc_hypre_type"] = "boomeramg"
    # opts[f"{option_prefix}pc_hypre_boomeramg_max_iter"] = 1
    # opts[f"{option_prefix}pc_hypre_boomeramg_cycle_type"] = "v"
    ksp.setFromOptions()
    # log.set_log_level(log.LogLevel.INFO)
    n, converged = solver.solve(w)
    assert (converged)
    print(f"Number of interations: {n:d}")

    # vtx_u = dolfinx.io.VTXWriter(  mesh.comm,   np_path + f"{  experiment_number}_{opt_step}_u.bp",   w.sub(0).collapse(),
    #                              engine="BP4")
    # vtx_u.write(0.0)
    # vtx_u.close()
    sol = Function(W)
    sol.x.array[:] = w.x.array
    return sol


def set_adjoint_equation(u, lam_2, x, h, u_d, q, u_r, adjoint_NS_variable_package, dx, ds, mesh, inlet_marker, W, bcu):
    w_adj, u_adj, p_adj, v_adj, pr_adj = adjoint_NS_variable_package
    a = viscosity * inner(grad(u_adj), grad(v_adj)) * dx
    c = (inner(u_adj, grad(v_adj) * u) + inner(u_adj, grad(u) * v_adj)) * dx
    b_form = inner(pr_adj, div(u_adj)) * dx
    div_ = inner(p_adj, div(v_adj)) * dx

    v_dot_n = dot(v_adj, FacetNormal(mesh))
    u_dot_n = dot(u, FacetNormal(mesh))

    cosh_quadr = dot(ufl.cosh(u_dot_n / delta), ufl.cosh(u_dot_n / delta))
    psi_d = 0.5 * (ufl.tanh(u_dot_n / delta) + u_dot_n / (delta * cosh_quadr) - 1)

    psi_delta = 0.5 * (u_dot_n * ufl.tanh(u_dot_n / delta) - u_dot_n + delta)
    d_delta = inner(psi_delta * v_adj, u_adj) * ds(inlet_marker)

    adj_extra_bt = 0.5 * (inner(v_dot_n * psi_d * (u), u_adj) * ds(inlet_marker) + d_delta)
    lhs_ = a + c + div_ - b_form #+ adj_extra_bt
    b = Function(W)
    b.x.array[:] = 0

    bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    zeros = np.zeros((*x.shape[:2], 1))
    new_points = np.concatenate((x, zeros), axis=-1)

    u_values_arr = []
    for k, buoy_points in enumerate(new_points):
        cells = []
        points_on_proc = []
        cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, buoy_points)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates,
                                                                   buoy_points)
        for i, point in enumerate(buoy_points):
            if len(colliding_cells.links(i)) > 0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])

        u_values = u.eval(points_on_proc, cells)
        gamma = (u_values - u_d[k, :, :] + lam_2[k, :, :])
        ps1 = scifem.PointSource(W.sub(0).sub(0), buoy_points, magnitude=h * gamma[:, 0])
        ps2 = scifem.PointSource(W.sub(0).sub(1), buoy_points, magnitude=h * gamma[:, 1])
        ps1.apply_to_vector(b)
        ps2.apply_to_vector(b)
        u_values_arr.append(u_values)

    # point_shape = new_points.shape
    # new_points = new_points.reshape(point_shape[0] * point_shape[1], point_shape[2])
    #
    # cells = []
    # points_on_proc = []
    # cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, new_points)
    # colliding_cells = dolfinx.geometry.compute_colliding_cells(  mesh, cell_candidates,
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
    # ps1 = scifem.PointSource(  W.sub(0).sub(0), new_points, magnitude=h * gamma[:, 0])
    # ps2 = scifem.PointSource(  W.sub(0).sub(1), new_points, magnitude=h * gamma[:, 1])
    # ps1.apply_to_vector(b)
    # ps2.apply_to_vector(b)

    lhs_form = form(lhs_)
    apply_lifting(b.x.petsc_vec, [lhs_form], [bcu])
    # b.x.scatter_reverse(dolfinx.la.InsertMode.add)
    dolfinx.fem.petsc.set_bc(b.x.petsc_vec, bcu)
    b.x.scatter_forward()
    A = dolfinx.fem.petsc.assemble_matrix(lhs_form, bcs=bcu)
    A.assemble()


    return A, b, u_values_arr


def adjoint_state_solving_step(u, lam_2, x, h, u_d, q, u_r, adjoint_NS_variable_package, dx, ds, mesh, inlet_marker, W,
                               bcu):
    print("solving adjoint NS")
    A, rhs, u_values = set_adjoint_equation(u, lam_2, x, h, u_d, q, u_r, adjoint_NS_variable_package, dx, ds, mesh,
                                               inlet_marker, W, bcu)
    solver = PETSc.KSP().create(mesh.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    solver.getPC().setFactorSolverType("mumps")
    # solver.setFromOptions()
    up = dolfinx.fem.Function(W)
    solver.solve(rhs.x.petsc_vec, up.x.petsc_vec)
    # up.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    up.x.scatter_forward()

    return up, u_values


def set_stokes_state(q, u_r, v_r, p_r, pr_r, mesh, dx, ds, inlet_marker, bcu):
    f = Constant(mesh, PETSc.ScalarType((0, 0)))
    a = viscosity * inner(grad(u_r), grad(v_r)) * dx
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


def set_adjoint_stokes(u, u_r_adj, v_r_adj, p_r_adj, pr_r_adj, mesh, dx, bcu):
    f = Constant(mesh, PETSc.ScalarType((0, 0)))
    a = viscosity * inner(grad(u_r_adj), grad(v_r_adj)) * dx
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


def solve_stokes_step(q, u_r, v_r, p_r, pr_r, mesh, dx, ds, inlet_marker, bcu, W):
    A, b, a, L = set_stokes_state(q, u_r, v_r, p_r, pr_r, mesh, dx, ds, inlet_marker, bcu)
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


def solve_adoint_stokes_step(u, u_r_adj, v_r_adj, p_r_adj, pr_r_adj, mesh, dx, bcu, W):
    A, b, a, L = set_adjoint_stokes(u, u_r_adj, v_r_adj, p_r_adj, pr_r_adj, mesh, dx, bcu)
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


def old_code(self):
    a = 1
    # for b in range(  K):
    #     points = np.array([x[b, :, :][:,0], x[b, :, :][:, 1], np.zeros(len(x[b, :, :][:,0]))]).T
    #     point_marker = 5
    #     num_entities_local =   mesh.topology.index_map(  mesh.topology.dim).size_local +   mesh.topology.index_map(  mesh.topology.dim).num_ghosts
    #     entities = np.arange(num_entities_local, dtype=np.int32)
    #     midpoint_tree = dolfinx.geometry.create_midpoint_tree(  mesh,   mesh.topology.dim, entities)
    #     #facet_indices = dolfinx.geometry.compute_closest_entity(bb_tree,midpoint_tree,  mesh, points)
    #     facet_indices = []
    #
    #     cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, points)
    #     facet_indices = np.unique(dolfinx.geometry.compute_colliding_cells(  mesh, cell_candidates, points).array)
    #
    #     facet_tags = dolfinx.mesh.meshtags(  mesh,   mesh.topology.dim-1, facet_indices, np.full_like(facet_indices, point_marker))
    #
    #
    #     ds = ufl.Measure("dS",domain=  mesh, subdomain_data=facet_tags, subdomain_id=point_marker)
    #
    #     # Term associated with points
    #     for l2 in lam_2[b, :, :]:
    #         F += l2*  v_adj * ds
    #     #line_integral_1 += lam_2[b, :, :] *   v_adj * ds

    # a_form = form(a)
    # K = create_matrix(a_form)
    # K.zeroEntries()
    # assemble_matrix(K, a_form, bcs=  bcu)
    # K.assemble()
    #
    # b_form = form(b)
    # Bt = create_matrix(b_form)
    # Bt.zeroEntries()
    # assemble_matrix(Bt, b_form, bcs=  bcus)
    # Bt.assemble()
    #
    # M_form = form(m)
    # M = create_matrix(M_form)
    # M.zeroEntries()
    # assemble_matrix(M, M_form, bcs=[  bcp])
    # M.assemble()
    #
    # c_form = form(c)
    # C = create_matrix(c_form)
    # C.zeroEntries()
    # assemble_matrix(C, c_form, bcs=  bcu)
    # C.assemble()
    #
    # f_form = form(f_)
    # f_vector = assemble_vector(f_form)
    # f_vector.assemble()
    # set_bc(f_vector, bcs=  bcus)
    # K, Bt, C, f, M, a_form, b_form =   set_state_equations(q)
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
    #     c = inner(dot(  u_k, nabla_grad(  u)),   v) *   dx
    #     c_form = form(c)
    #     C = assemble_matrix(c_form, bcs=  bcus)
    #     C.assemble()
    #     C_np = petsc_matrix_to_numpy(C)
    #     lhs1 = K_np +   alpha * C_np
    #     rhs1 = f_np + (  alpha - 1) * C_np @   u_k.x.array - Bt_np @   p_k.x.array
    #
    #     u_hat_s = sc.linalg.solve(lhs1, rhs1)
    #
    #     print(K_np[m,n])
    #     intr =   u_k.x.array
    #     from IPython import embed
    #     embed()
    #     delta_p = sc.linalg.solve(lhs2, M_inv @ (Bt_np.T @ u_hat_s))
    #
    #     u_hat = sc.linalg.solve(K_np, K_np @ u_hat_s - Bt_np @ delta_p)
    #
    #       p_s.x.array[:] = delta_p +   p_k.x.array
    #
    #       u_s.x.array[:] =   gamma * u_hat + (1 -   gamma) *   u_k.x.array
    #
    #     # p_old =   p_s.x.array[:]
    #     # u_old =   u_s.x.array[:]
    #
    #     u_err = np.linalg.norm(  u_s.x.array -   u_k.x.array)
    #     p_err = np.linalg.norm(  p_s.x.array -   p_k.x.array)
    #     print(i)
    #     print(u_err)
    #     print(p_err)
    #     if u_err < 1e-6 and p_err < 1e-6:
    #         break
    #
    #       p_k.x.array[:] =   p_s.x.array
    #       u_k.x.array[:] =   u_s.x.array
    #
    #   save_vector_fields()
    # B = PETSc.Mat().createTranspose(Bt)
    # solver1 = PETSc.KSP().create(MPI.COMM_WORLD)
    # solver1.setOperators(K +   alpha * C)
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
    # u_hat2 = Function(  U)
    #
    # L1 = form(f + (  alpha - 1) * inner(dot(  u_k, nabla_grad(  u_k)),   v) *   dx + inner(div(  v),   p_k) *   dx)
    # b1 = create_vector(L1)
    #
    # L2 = form(-inner(div(  u_hat),   pr) *   dx)
    # b2 = create_vector(L2)
    #
    # L3 = form(  viscosity * inner(grad(  u_hat) + grad(  u_hat).T, grad(  v)) *   dx + inner(div(  v),   p_hat) *   dx)
    # b3 = create_vector(L3)
    #
    # from IPython import embed
    # embed()
    # for i in range(100):
    #     assemble_vector(b1, L1)
    #     apply_lifting(b1, [a_form], [  bcus])
    #     b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    #     set_bc(b1,   bcus)
    #     solver1.solve(b1,   u_hat.x.petsc_vec)
    #       u_hat.x.scatter_forward()
    #
    #     # u_.x.petsc_vec.axpy(1,   u_hat.x.petsc_vec)
    #     # u_.x.scatter_forward()
    #
    #     assemble_vector(b2, L2)
    #     apply_lifting(b2, [b_form], [  bcus])
    #     b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    #     set_bc(b2,   bcus)
    #     solver2.solve(b2,   p_hat.x.petsc_vec)
    #       p_hat.x.scatter_forward()
    #
    #     # p_.x.petsc_vec.axpy(1,   p_hat.x.petsc_vec)
    #     # p_.x.scatter_forward()
    #
    #     assemble_vector(b3, L3)
    #     apply_lifting(b3, [a_form], [  bcus])
    #     b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    #     set_bc(b3,   bcus)
    #     solver3.solve(b3, u_hat2.x.petsc_vec)
    #     u_hat2.x.scatter_forward()
    #
    #       p_s.x.array[:] =   p_hat.x.array +   p_k.x.array
    #       u_s.x.array[:] =   gamma * u_hat2.x.array + (1-  gamma) *   u_k.x.array
    #
    #     u_err = np.linalg.norm(  u_s.x.array -   u_k.x.array)
    #     p_err = np.linalg.norm(  p_s.x.array -   p_k.x.array)
    #     print(u_err)
    #     print(p_err)
    #
    #       u_k.x.array[:] =   u_s.x.array
    #       p_k.x.array[:] =   p_s.x.array
    #       u_k.x.scatter_forward()
    #       p_k.x.scatter_forward()
    #     # from IPython import embed
    #     # embed()
    #
    #     print(i)
    return a
