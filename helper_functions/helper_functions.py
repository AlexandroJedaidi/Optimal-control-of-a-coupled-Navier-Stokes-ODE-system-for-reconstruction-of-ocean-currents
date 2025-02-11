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
import solver_classes.Navier_stokes_solver
import solver_classes.ODE_solver


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
    dq.x.array[:] = 0.01  # q_func.x.array[:]
    h_ = np.array([10 ** -(i + 1) for i in range(15)])

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


    # ei = np.zeros_like(q_func.x.array[:].copy())
    # grad_J_FD,_ = Function(W).split()
    # hh = 1e-6
    # for i in range(ei.shape[0]):
    #     ei = np.zeros_like(q_func.x.array[:].copy())
    #     ei[i] = 1
    #
    #     qhat = Function(W)  # .split()
    #     qhat.sub(0).x.array[:] = q_func.x.array[:] + hh * ei
    #     # qhat = q_func + h_i.item() * dq
    #     # qhat_expr = dolfinx.fem.Expression(qhat, U.element.interpolation_points())
    #     # qhat_fct = dolfinx.fem.Function(U)
    #     # qhat_fct.interpolate(qhat_expr)
    #     w_ = NS_instance.state_solving_step(qhat.sub(0), u_ref, opt_iter)
    #     u_, p_ = w_.split()
    #     qnorm = form(dot(qhat.sub(0), qhat.sub(0)) * ds_(inlet_marker))
    #     E_ = assemble_scalar(qnorm)
    #     x_ = ODE_instance.ode_solving_step(u_)
    #     u_values_ = np.array(evalutate_fuct(u_, x_, mesh))
    #     dJh = 0.5 * np.sum(np.sum(h * (np.linalg.norm(np.array(u_values_) - u_d, axis=2) ** 2), axis=1)) + (
    #             alpha / 2) * E
    #     grad_J_FD.x.array[i] = dJh

    dJ = []
    for h_i in h_:
        qhat = Function(W)  # .split()
        qhat.sub(0).x.array[:] = q_func.x.array[:] + h_i.item() * dq.x.array[:]
        # qhat = q_func + h_i.item() * dq
        # qhat_expr = dolfinx.fem.Expression(qhat, U.element.interpolation_points())
        # qhat_fct = dolfinx.fem.Function(U)
        # qhat_fct.interpolate(qhat_expr)
        w_ = NS_instance.state_solving_step(qhat.sub(0), u_ref, opt_iter)
        u_, p_ = w_.split()
        qnorm = form(dot(qhat.sub(0), qhat.sub(0)) * ds_(inlet_marker))
        E_ = assemble_scalar(qnorm)
        if ODE_instance is not None:
            x_ = ODE_instance.ode_solving_step(u_)
            u_values_ = np.array(evalutate_fuct(u_, x_, mesh))
            dJh = 0.5 * np.sum(np.sum(h * (np.linalg.norm(np.array(u_values_) - u_d, axis=2) ** 2), axis=1)) + (
                    alpha / 2) * E
        else:
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
            plt.quiver(x_, y_, u_vec, v_vec, 1, angles="xy", scale_units="xy", scale=0.6, cmap="viridis")
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
