import ufl
from mpi4py import MPI
import json
from dolfinx.fem import (Constant, Function, functionspace,
                         assemble_scalar, dirichletbc, form, locate_dofs_topological, set_bc)
from ufl import (FacetNormal, Identity, Measure, TestFunction, TrialFunction,
                 as_vector, div, dot, inner, lhs, grad, nabla_grad, rhs, sym, system, SpatialCoordinate, inv,
                 sqrt, transpose, tr)
import numpy as np
import dolfinx
import matplotlib.pyplot as plt
from basix.ufl import element, mixed_element
from dolfinx import mesh, fem, io
import solver_classes.Navier_stokes_solver as NS
import solver_classes.ODE_solver
from test_pipelines.stokes_helper import solve_stokes
from solver_classes import multiphenicsx_NS_solver


def evalutate_fuct(fct, points, mesh):
    bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    zeros = np.zeros((*points.shape[:2], 1))
    new_points = np.concatenate((points, zeros), axis=-1)

    point_shape = new_points.shape
    # new_points = new_points.reshape(point_shape[0] * point_shape[1], point_shape[2])
    u_arr = []
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
        u_arr.append(fct.eval(points_on_proc, cells))

    return u_arr


def evalutate_general_fuct(fct, points, mesh):
    bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    cells = []
    points_on_proc = []
    cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates,
                                                               points)
    for i, point in enumerate(points):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
    u_arr = fct.eval(points_on_proc, cells)
    return u_arr


def test_gradient(lam_, q_func, u_red, x_red, opt_iter, u_ref, ft, W, alpha, inlet_marker, u_d, h, NS_instance,
                  ODE_instance, np_path, mesh):
    dx_ = ufl.Measure('dx', domain=mesh)
    ds_ = ufl.Measure("ds", subdomain_data=ft)
    dq, _ = Function(W).split()
    dq.x.array[:] = q_func.x.array[:]
    # h_ = np.array([0.1*i for i in range(1,10)][::-1])
    h_ = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2][::-1])

    grad_j_funcp = Function(W)
    grad_j_func, _ = grad_j_funcp.split()
    grad_j_func.x.array[:] = alpha * q_func.x.array[:] - lam_.x.array[:]
    red_grad = form(dot(grad_j_func, dq) * ds_(inlet_marker))
    J_grad_red = assemble_scalar(red_grad)

    qnorm = form(dot(q_func, q_func) * ds_(inlet_marker))
    E = assemble_scalar(qnorm)

    if u_d is not None or x_red is not None:
        u_values_red = np.array(evalutate_fuct(u_red, x_red, mesh))
        J_red = 0.5 * np.sum(np.sum(h * (np.linalg.norm(np.array(u_values_red) - u_d, axis=2) ** 2), axis=1)) + (
                alpha / 2) * E
    else:
        E_u = assemble_scalar(form(dot(u_red, u_red) * dx_))
        J_red = 0.5 * E_u + (alpha / 2) * E

    dJ = []
    for h_i in h_:
        qhat = Function(W)  # .split()
        qhat.sub(0).x.array[:] = q_func.x.array[:] + h_i.item() * dq.x.array[:]
        qnorm = form(dot(qhat.sub(0), qhat.sub(0)) * ds_(inlet_marker))
        E_ = assemble_scalar(qnorm)

        if ODE_instance is not None:
            w_ = NS_instance.state_solving_step(qhat.sub(0), u_ref, opt_iter)
            u_, p_ = w_.split()
            x_ = ODE_instance.ode_solving_step(u_)
            u_values_ = np.array(evalutate_fuct(u_, x_, mesh))
            dJh = 0.5 * np.sum(np.sum(h * (np.linalg.norm(np.array(u_values_) - u_d, axis=2) ** 2), axis=1)) + (
                    alpha / 2) * E_
        else:
            w_ = NS_instance(qhat.sub(0))
            u_, p_ = w_.split()
            u_norm = form(dot(u_, u_) * dx_)
            E_u_ = assemble_scalar(u_norm)
            dJh = 0.5 * E_u_ + (alpha / 2) * E_

        approx_J = (dJh - J_red) / h_i.item()
        dJ.append(approx_J)

    with open(np_path + f"grad_J_error_{opt_iter}.txt", "w") as text_file:
        text_file.write("reduced Gradient j \t \t approximated gradient J \t Error \t \t \t h_i \n")
        for i, h_i in enumerate(h_[:-1]):
            text_file.write(f" {J_grad_red} \t {dJ[i]} \t {abs(dJ[i] - J_grad_red)} \t {h_i} \n")

    return dJ[-1]


def test_gradient_centered_finite_differences_NS(lam_, q_func, u_red, x_red, opt_iter, u_ref, ft, W, alpha,
                                                 inlet_marker, u_d, h, ODE_instance, np_path, mesh, ds, primal_NS_variable_package, dx, bcu, vis):
    dq, _ = Function(W).split()

    def dq_in(x):
        values = np.zeros((2, x.shape[1]))
        values[0, :] = np.where(np.isclose(x[0, :], 0.0),
                                x[1] * (1 - x[1]),
                                0.0)
        # values[1, :] = np.where(np.isclose(x[0, :], 0.0),
        #                         0.0,
        #                         0.0)
        return values

    dq.interpolate(dq_in)
    h_ = np.array([10 ** (-i) for i in range(2, 11)])

    grad_j_funcp = Function(W)
    grad_j_func, _ = grad_j_funcp.split()
    grad_j_func.x.array[:] = alpha * q_func.x.array[:] + lam_.x.array[:]
    red_grad = form(dot(grad_j_func, dq) * ds(inlet_marker))
    J_grad_red = assemble_scalar(red_grad)

    dJ = []
    dJ_one_sided = []

    def J(u, f_):
        return 0.5 * np.sum(np.sum(h * (np.linalg.norm(u - u_d, axis=2) ** 2), axis=1)) + assemble_scalar(
            form(alpha * 0.5 * inner(f_, f_) * ds(inlet_marker)))

    u_values_0 = np.array(evalutate_fuct(u_red, x_red, mesh))
    J0 = J(u_values_0, q_func)
    for h_i in h_:
        print(h_i)
        qhatp, _ = Function(W).split()
        qhatp.x.array[:] = q_func.x.array[:] + h_i.item() * dq.x.array[:]
        w_p = multiphenicsx_NS_solver.run_monolithic(mesh, vis, bcu, qhatp, ds, 2)
        # w_p = NS.state_solving_step(primal_NS_variable_package, qhatp, None, opt_iter,mesh, dx, ds, inlet_marker, bcu, W)
        u_p, p_p = w_p.split()
        x_p = ODE_instance.ode_solving_step(u_p)
        u_values_ = np.array(evalutate_fuct(u_p, x_p, mesh))
        dJhp = J(u_values_, qhatp)

        qhatm, _ = Function(W).split()
        qhatm.x.array[:] = q_func.x.array[:] - h_i.item() * dq.x.array[:]
        w_ = multiphenicsx_NS_solver.run_monolithic(mesh, vis, bcu, qhatm, ds, 2)
        # w_ = NS.state_solving_step(primal_NS_variable_package, qhatm, None, opt_iter,mesh, dx, ds, inlet_marker, bcu, W)
        u_, p_ = w_.split()
        x_ = ODE_instance.ode_solving_step(u_)
        u_values_m = np.array(evalutate_fuct(u_, x_, mesh))
        dJhm = J(u_values_m, qhatm)

        approx_J = (dJhp - dJhm) / (2 * h_i.item())
        approx_J_one_side = (dJhp - J0) / (h_i.item())
        dJ.append(approx_J)
        dJ_one_sided.append(approx_J_one_side)

    with open(np_path + f"grad_J_error_two_side_{opt_iter}.txt", "w") as text_file:
        text_file.write("reduced Gradient j \t \t approximated gradient J \t Error \t \t \t h_i \n")
        for i, h_i in enumerate(h_[:-1]):
            text_file.write(f" {J_grad_red} \t {dJ[i]} \t {abs(dJ[i] - J_grad_red)} \t {h_i} \n")

    with open(np_path + f"grad_J_error_one_side_{opt_iter}.txt", "w") as text_file:
        text_file.write(
            "calculated FD inner products \t  reduced Gradient j \t \t approximated gradient J \t Error \t \t \t h_i \n")
        for i, h_i in enumerate(h_[:-1]):
            text_file.write(f" {J_grad_red} \t {dJ_one_sided[i]} \t {abs(dJ_one_sided[i] - J_grad_red)} \t {h_i} \n")

    return dJ[-1]


def test_gradient_centered_finite_differences(z, q, opt_iter, ds, W, alpha, np_path, mesh, lhs, v, bc, u_d, u, top,
                                              bottom, left, right):
    dx_ = ufl.Measure('dx', domain=mesh)
    dq, _ = Function(W).split()

    def dq_in(x):
        values = np.zeros((2, x.shape[1]))
        if left:
            values[0, :] = np.where(np.isclose(x[0, :], 0.0),
                                    x[1] * (1 - x[1]),
                                    0.0)
            values[1, :] = np.where(np.isclose(x[0, :], 0.0),
                                    0,
                                    0.0)
        if right:
            values[0, :] = np.where(np.isclose(x[0, :], 1.0),
                                    x[1],
                                    0.0)
            values[1, :] = np.where(np.isclose(x[0, :], 1.0),
                                    x[1],
                                    0.0)
        if top:
            values[0, :] = np.where(np.isclose(x[1, :], 1.0),
                                    x[0],
                                    0.0)
            # np.logical_and(np.where(np.isclose(x[0, :], 0.5), 10000, 0.0),np.where(np.isclose(x[1, :], 0.5), 10000, 0.0)))
            values[1, :] = np.where(np.isclose(x[1, :], 1.0),
                                    x[0],
                                    0.0)
        if bottom:
            values[0, :] = np.where(np.isclose(x[1, :], 0.0),
                                    x[0],
                                    0.0)
            # np.logical_and(np.where(np.isclose(x[0, :], 0.5), 10000, 0.0),np.where(np.isclose(x[1, :], 0.5), 10000, 0.0)))
            values[1, :] = np.where(np.isclose(x[1, :], 0.0),
                                    x[0],
                                    0.0)
        return values

    dq.interpolate(dq_in)

    h_ = np.array([10 ** (-i) for i in range(-1, 12)])

    grad_j_funcp = Function(W)
    grad_j_func, _ = grad_j_funcp.split()
    grad_j_func.x.array[:] = alpha * q.x.array[:] + z.x.array[:]
    red_grad = form(dot(grad_j_func, dq) * ds(2))
    J_grad_red = assemble_scalar(red_grad)

    dJ = []
    dJ_one_side = []

    def J(u, f_):
        return assemble_scalar(form(0.5 * inner(u - u_d, u - u_d) * dx_ + alpha * 0.5 * inner(f_, f_) * ds(2)))

    J0 = J(u, q)
    for h_i in h_:
        print(h_i)
        qhatp, _ = Function(W).split()
        qhatp.x.array[:] = q.x.array[:] + h_i.item() * dq.x.array[:]

        problem = dolfinx.fem.petsc.LinearProblem(lhs, inner(qhatp, v) * ds(2), bcs=bc, petsc_options={
            "ksp_type": "preonly",
            # "pc_type": "lu",
            # "pc_factor_mat_solver_type": "superlu_dist",
        })
        w_p = problem.solve()
        u_p, p_p = w_p.split()
        J_p = J(u_p, qhatp)

        qhatm, _ = Function(W).split()
        qhatm.x.array[:] = q.x.array[:] - h_i.item() * dq.x.array[:]

        problem_m = dolfinx.fem.petsc.LinearProblem(lhs, inner(qhatm, v) * ds(2), bcs=bc, petsc_options={
            "ksp_type": "preonly",
            # "pc_type": "lu",
            # "pc_factor_mat_solver_type": "superlu_dist",
        })
        w_ = problem_m.solve()
        u_, p_ = w_.split()
        J_m = J(u_, qhatm)

        approx_J = (J_p - J_m) / (2 * h_i.item())
        approx_J_one_side = (J_p - J0) / (h_i.item())
        dJ.append(approx_J)
        dJ_one_side.append(approx_J_one_side)

    with open(np_path + f"grad_J_error_two_side_{opt_iter}.txt", "w") as text_file:
        text_file.write("reduced Gradient j \t \t approximated gradient J \t Error \t \t \t h_i \n")
        for i, h_i in enumerate(h_[:-1]):
            text_file.write(f" {J_grad_red} \t {dJ[i]} \t {abs(dJ[i] - J_grad_red)} \t {h_i} \n")

    with open(np_path + f"grad_J_error_one_side_{opt_iter}.txt", "w") as text_file:
        text_file.write("reduced Gradient j \t \t approximated gradient J \t Error \t \t \t h_i \n")
        for i, h_i in enumerate(h_[:-1]):
            text_file.write(f" {J_grad_red} \t {dJ_one_side[i]} \t {abs(dJ_one_side[i] - J_grad_red)} \t {h_i} \n")

    with open(np_path + f"grad_J_FD_comparison_{opt_iter}.txt", "w") as text_file:
        text_file.write("right side Gradient approx \t \t centered gradient approx \t Error \t \t \t h_i \n")
        for i, h_i in enumerate(h_[:-1]):
            text_file.write(f" {dJ_one_side[i]} \t {dJ[i]} \t {abs(dJ[i] - dJ_one_side[i])} \t {h_i} \n")
    return dJ[-1]


def test_gradient_centered_finite_differences_on_rhs_control(z, f, opt_iter, W, alpha, np_path, mesh, lhs, v, bc, u,
                                                             u_d):
    dx_ = ufl.Measure('dx', domain=mesh)
    df, _ = Function(W).split()

    def dq_in(x):
        values = np.zeros((2, x.shape[1]))
        values[0, :] = x[1]
        values[1, :] = x[1]
        return values

    df.interpolate(dq_in)

    h_ = np.array([10 ** (-i) for i in range(-1, 13)])

    grad_j_funcp = Function(W)
    grad_j_func, _ = grad_j_funcp.split()
    grad_j_func.x.array[:] = alpha * f.x.array[:] + z.x.array[:]
    red_grad = form(dot(grad_j_func, df) * dx_)
    J_grad_red = assemble_scalar(red_grad)

    dJ = []
    dJ_one_side = []

    def J(u, f_):
        return assemble_scalar(form(0.5 * inner(u - u_d, u - u_d) * dx_ + alpha * 0.5 * inner(f_, f_) * dx_))

    J0 = J(u, f)
    for h_i in h_:
        print(h_i)
        fhatpp = Function(W)
        fhatp, _ = fhatpp.split()
        fhatp.x.array[:] = f.x.array[:] + h_i.item() * df.x.array[:]
        problem = dolfinx.fem.petsc.LinearProblem(lhs, inner(fhatp, v) * dx_, bcs=[bc])
        w_p = problem.solve()
        u_p, p_p = w_p.split()
        J_p = J(u_p, fhatp)

        fhatm, _ = Function(W).split()
        fhatm.x.array[:] = f.x.array[:] - h_i.item() * df.x.array[:]

        problem_m = dolfinx.fem.petsc.LinearProblem(lhs, inner(fhatm, v) * dx_, bcs=[bc])
        w_m = problem_m.solve()
        u_m, p_m = w_m.split()
        J_m = J(u_m, fhatm)

        approx_J = (J_p - J_m) / (2 * h_i.item())
        one_side_J = (J_p - J0) / h_i.item()
        dJ.append(approx_J)
        dJ_one_side.append(one_side_J)

    with open(np_path + f"grad_J_error_two_side_{opt_iter}.txt", "w") as text_file:
        text_file.write("reduced Gradient j \t \t approximated gradient J \t Error \t \t \t h_i \n")
        for i, h_i in enumerate(h_[:-1]):
            text_file.write(f" {J_grad_red} \t {dJ[i]} \t {abs(dJ[i] - J_grad_red)} \t {h_i} \n")

    with open(np_path + f"grad_J_error_one_side_{opt_iter}.txt", "w") as text_file:
        text_file.write("reduced Gradient j \t \t approximated gradient J \t Error \t \t \t h_i \n")
        for i, h_i in enumerate(h_[:-1]):
            text_file.write(f" {J_grad_red} \t {dJ_one_side[i]} \t {abs(dJ_one_side[i] - J_grad_red)} \t {h_i} \n")

    return dJ[-1]


def eval_vector_field(grid, coordinates, func, func_name, path, vector_field=True):
    if vector_field:
        bb_tree = dolfinx.geometry.bb_tree(grid, grid.topology.dim)
        func_eval = []
        for point in coordinates:
            cells = []
            points_on_proc = []
            cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, point)
            colliding_cells = dolfinx.geometry.compute_colliding_cells(grid, cell_candidates, point)
            if len(colliding_cells.links(0)) > 0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(0)[0])
            points_on_proc = np.array(points_on_proc, dtype=np.float64)
            if len(colliding_cells) < 0:
                print("no colliding cells")
                from IPython import embed
                embed()
            func_eval.append(func.eval(points_on_proc, cells))
        y_axis = []
        for i, coord in enumerate(func_eval):
            x_ = coordinates[i][0]
            y_ = coordinates[i][1]
            y_axis.append(y_)
            u_vec = func_eval[i][0]
            v_vec = func_eval[i][1]
            plt.quiver(x_, y_, u_vec, v_vec, 1, angles="xy", scale_units="xy", scale=1, cmap="viridis")
        plt.savefig(f"{path}/{func_name}_vectorfield.png")
        plt.clf()
    else:
        coordinates = np.sort(coordinates, axis=0)
        bb_tree = dolfinx.geometry.bb_tree(grid, grid.topology.dim)
        func_eval = []
        for point in coordinates:
            cells = []
            points_on_proc = []
            cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, point)
            colliding_cells = dolfinx.geometry.compute_colliding_cells(grid, cell_candidates, point)
            if len(colliding_cells.links(0)) > 0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(0)[0])
            points_on_proc = np.array(points_on_proc, dtype=np.float64)
            if len(colliding_cells) < 0:
                print("no colliding cells")
                from IPython import embed
                embed()
            func_eval.append(func.eval(points_on_proc, cells))
        y_axis = []
        for i, coord in enumerate(func_eval):
            y_ = coordinates[i][1]
            y_axis.append(y_)
        plt.plot(y_axis, [item[0] for item in func_eval])
        plt.savefig(f"{path}/{func_name}_flow_profile.png")
        plt.clf()


def testing_test_gradient_func():
    domain = mesh.create_rectangle(MPI.COMM_WORLD, [(0.0, 0.0), (1.0, 1.0)],
                                   (3, 3), mesh.CellType.quadrilateral)
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim - 1, tdim)
    dx_ = ufl.Measure('dx', domain=domain)
    U_el = element("Lagrange", domain.basix_cell(), 2, shape=(domain.geometry.dim,))
    P_el = element("Lagrange", domain.basix_cell(), 1)
    W_el = mixed_element([U_el, P_el])
    W = functionspace(domain, W_el)
    U, _ = W.sub(0).collapse()
    u = dolfinx.fem.Function(U)

    def u_values(x):
        values = np.zeros((2, x.shape[1]))
        # values[0, :] = np.where(np.isclose(x[0, :], 0.0), 1.0, np.where(np.isclose(x[0, :], 2.0), 1.0, 0.0))  # x-component
        # values[1, :] = np.where(np.isclose(x[0, :], 0.0), 0.0, 0.0)  # y-component
        values[0, :] = x[0, :] ** 2 + x[1, :] ** 2
        return values

    u.interpolate(u_values)

    du = dolfinx.fem.Function(U)
    du.x.array[:] = u.x.array[:]

    grad_u = ufl.grad(u)
    U_grad_fp = dolfinx.fem.functionspace(domain,
                                          ("Lagrange", 1, (domain.geometry.dim, domain.geometry.dim)))
    u_grad_expr = dolfinx.fem.Expression(grad_u, U_grad_fp.element.interpolation_points())
    u_grad_fct = dolfinx.fem.Function(U_grad_fp)
    u_grad_fct.interpolate(u_grad_expr)
    exact_grad = assemble_scalar(form(inner(grad_u, grad_u) * dx_))

    h_ = np.array([10 ** -(i + 1) for i in range(15)])

    dJ = []
    for h_i in h_:
        uhat = Function(U)
        uhat.x.array[:] = u.x.array[:] + h_i.item() * du.x.array[:]

        approx_J = (uhat - u) / h_i.item()
        dJ.append(approx_J)

    with open(f"helper_functions/test_results/grad_J_error_{0}.txt", "w") as text_file:
        text_file.write("reduced Gradient j \t \t approximated gradient J \t Error \t \t \t h_i \n")
        for i, h_i in enumerate(h_[:-1]):
            text_file.write(f" {exact_grad} \t {dJ[i]} \t {abs(dJ[i] - exact_grad)} \t {h_i} \n")

# testing_test_gradient_func()
