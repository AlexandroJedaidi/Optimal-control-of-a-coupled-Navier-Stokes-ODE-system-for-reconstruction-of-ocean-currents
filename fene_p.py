import dolfinx
import ufl
from ufl import (FacetNormal, Identity, TestFunction, TrialFunction,
                 div, ds, dx, inner, lhs, nabla_grad, rhs, sym)
import basix
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import petsc4py
import dolfinx.fem.petsc
from dolfinx.nls.petsc import NewtonSolver
import pyvista
import os
import matplotlib.pyplot as plt
from slepc4py import SLEPc


def function_space(domain):
    # function space
    s_el_dim = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1, shape=(2, 2))
    V = dolfinx.fem.functionspace(domain, s_el_dim)

    # Test function definition
    sigma = dolfinx.fem.Function(V)
    phi = ufl.TestFunction(V)
    return V, sigma, phi


def boundary_conditions(domain, V, x):
    # boundary conditions
    s_D = dolfinx.fem.Function(V)
    s_D.x.petsc_vec.set(0.0)
    fdim = domain.topology.dim - 1
    boundary = lambda x: np.logical_or(np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[1], 1.0)),
                                       np.logical_or(np.isclose(x[0], 1.0), np.isclose(x[1], 0.0)))
    boundary_facets = dolfinx.mesh.locate_entities_boundary(domain, fdim, boundary)
    bc = dolfinx.fem.dirichletbc(s_D, dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets))
    return bc


def vector_field(x, mesh):
    # vector field definition u(x,y)=(y-y²,0)
    v_cg2 = basix.ufl.element("Lagrange", mesh.topology.cell_name(), 1, shape=(mesh.geometry.dim,))
    V = dolfinx.fem.functionspace(mesh, v_cg2)
    f = dolfinx.fem.Function(V)
    f.x.petsc_vec.set(0.0)
    u1 = x[1] - x[1] ** 2
    u2 = 0
    return [u1, u2]


def A(sigma, b):
    return 1 / (1 - ufl.tr(sigma) / b)


def problem_definition(sigma, sigma_n, dt, vector_field1, vector_field2, phi, b, Wi, alpha):
    # Problem definition
    # div_tau = 100 * ufl.dot(div((A(sigma, b)) * sigma - Identity(2)), v) / Wi
    t1 = (ufl.tr((sigma - sigma_n) / dt * ufl.transpose(phi))) * dx
    # t2 = ufl.tr(ufl.nabla_div(ufl.as_vector([vector_field1, vector_field2]))*sigma * ufl.transpose(phi)) * dx
    nabla_term = vector_field1 * sigma.dx(0) + vector_field2 * sigma.dx(1)
    t2 = (ufl.tr(nabla_term * ufl.transpose(phi))) * dx
    t3 = (ufl.tr(ufl.grad(ufl.as_vector([vector_field1, vector_field2])) * sigma * ufl.transpose(phi))) * dx
    t4 = (ufl.tr(
        sigma * ufl.transpose(ufl.grad(ufl.as_vector([vector_field1, vector_field2]))) * ufl.transpose(phi))) * dx
    t5 = (A(sigma, b) / (Wi) * ufl.tr(sigma * ufl.transpose(phi)) - ufl.tr(phi)) * dx
    triple_dot = ufl.inner(ufl.grad(sigma), ufl.grad(phi))
    extra_diffusion = (alpha * triple_dot) * dx
    F_new = t1 + t2 - t3 - t4 + t5 + extra_diffusion
    return F_new


def solution_initialization(steps, V):
    sigma_n = dolfinx.fem.Function(V)
    #sigma_n.vector.set(0.0)

    sigma_11_solution_data = []
    sigma_12_solution_data = []
    sigma_21_solution_data = []
    sigma_22_solution_data = []
    sigma_n_array = sigma_n.x.array
    n = int(sigma_n_array.shape[0] / 4)
    sigma_11_solution_data.append([sigma_n_array[4 * k] for k in range(n)])
    sigma_12_solution_data.append([sigma_n_array[4 * k + 1] for k in range(n)])
    sigma_21_solution_data.append([sigma_n_array[4 * k + 2] for k in range(n)])
    sigma_22_solution_data.append([sigma_n_array[4 * k + 3] for k in range(n)])
    time_values_data = np.zeros(steps + 1)
    return sigma_n, sigma_11_solution_data, sigma_12_solution_data, sigma_21_solution_data, sigma_22_solution_data, time_values_data


def solve(sigma, sigma_n, dt, vector_field1, vector_field2, bc, phi, b, Wi, alpha, cond):
    F = problem_definition(sigma, sigma_n, dt, vector_field1, vector_field2, phi, b, Wi, alpha)
    problem = dolfinx.fem.petsc.NonlinearProblem(F, sigma)  # , bcs=[bc])
    newton = True
    try:
        solver = NewtonSolver(MPI.COMM_WORLD, problem)
        solver.convergence_criterion = "incremental"
        solver.report = True
        solver.rtol = 1e-4
        solver.max_it = 1000
        if not newton:
            ksp = solver.krylov_solver
            opts = PETSc.Options()
            option_prefix = ksp.getOptionsPrefix()
            opts[f"{option_prefix}ksp_type"] = "cg"
            opts[f"{option_prefix}pc_type"] = "gamg"
            opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
            ksp.setFromOptions()
        # dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
        n, converged = solver.solve(sigma)
        # try:
        #     t = SLEPc.EPS.Which.LARGEST_MAGNITUDE
        #     eig_max = test(solver.A,t)
        #     f = SLEPc.EPS.Which.SMALLEST_MAGNITUDE
        #     eig_min = test(solver.A,f)
        #     cond.append(eig_max[-1]/eig_min[-1])
        # except:
        #     print("EW crash")
        # print(cond)
        # assert (converged)
        # print(f"Number of iterations: {n:d}")
        return False
    except:
        return True


def save_solutions(sigma, sigma_11_solution_data, sigma_12_solution_data, sigma_21_solution_data,
                   sigma_22_solution_data, time_values_data, k, t):
    sigma_array = sigma.x.array
    n = int(sigma_array.shape[0] / 4)
    sigma_11_solution_data.append([sigma_array[4 * k] for k in range(n)])
    sigma_12_solution_data.append([sigma_array[4 * k + 1] for k in range(n)])
    sigma_21_solution_data.append([sigma_array[4 * k + 2] for k in range(n)])
    sigma_22_solution_data.append([sigma_array[4 * k + 3] for k in range(n)])
    time_values_data[k + 1] = t


def plotting(sigma_sol, V):
    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
    # x = np.concatenate([x[:,0:2],sigma_11.reshape(-1,1)],axis=1)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    grid.point_data["sigma"] = sigma_sol
    grid.set_active_scalars("sigma")
    u_plotter = pyvista.Plotter()
    u_plotter.add_mesh(grid, show_edges=True)
    u_plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        u_plotter.show()


def plotting_gif(sigma_list, V):
    plotter = pyvista.Plotter()
    plotter.open_gif("sigma_11.gif", fps=30)
    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
    # x = np.concatenate([x[:,0:2],sigma_11.reshape(-1,1)],axis=1)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    grid.point_data["sigma"] = sigma_list[0]
    warped = grid.warp_by_scalar("sigma", factor=0.5)
    plotter.add_mesh(warped, show_edges=True, clim=[np.min(sigma_list), np.max(sigma_list)])
    # grid.set_active_scalars("sigma")
    for sigma_sol in sigma_list:
        new_warped = grid.warp_by_scalar("sigma", factor=0.1)
        warped.points[:, :] = new_warped.points
        warped.point_data["sigma"][:] = sigma_sol
        plotter.write_frame()
    # plotter.view_xy()
    plotter.close()


def pipeline():
    experiment_number = 11
    np_path = f'results/fp/experiments/arrays/{experiment_number}/'
    plot_path = f"results/fp/experiments/plots/{experiment_number}/"
    os.mkdir(np_path)
    os.mkdir(plot_path)

    Lx = 1.0
    Ly = 1.0
    Nx = 50
    Ny = 50
    b, Wi, alpha = 30, 0.50, 0.1
    # mesh
    comm_t = MPI.COMM_WORLD
    domain = dolfinx.mesh.create_rectangle(comm_t, [np.array([0., 0.]), np.array([Lx, Ly])],
                                           [Nx, Ny], dolfinx.mesh.CellType.triangle)
    V, sigma, phi = function_space(domain)
    x = ufl.SpatialCoordinate(domain)
    bc = boundary_conditions(domain, V, x)
    u = vector_field(x, domain)
    vector_field1 = u[0]
    vector_field2 = u[1]
    steps = 10
    T = 0.1
    t = t0 = 0
    num_it = int((T - t0) / steps)
    dt = T / steps

    with open(np_path + "variables.txt", "w") as text_file:
        text_file.write("vector field1: %s \n" % str(u[0]))
        text_file.write("vector field2: %s \n" % str(u[1]))
        text_file.write("domain length Lx: %s \n" % str(Lx))
        text_file.write("domain length Ly: %s \n" % str(Ly))
        text_file.write("domain size Nx: %s \n" % str(Nx))
        text_file.write("domain size Ny: %s \n" % str(Ny))
        text_file.write("Weissenberg Number: %s \n" % Wi)
        text_file.write("Max Extension: %s \n" % b)
        text_file.write("dt: %s \n" % steps)
        text_file.write("t0: %s \n" % t0)
        text_file.write("T: %s \n" % T)
        text_file.write("alpha: %s" % alpha)

    _, tau_11_solution_data, tau_12_solution_data, tau_21_solution_data, tau_22_solution_data, time_values_data = solution_initialization(
        steps, V)

    sigma_n, sigma_11_solution_data, sigma_12_solution_data, sigma_21_solution_data, sigma_22_solution_data, time_values_data = solution_initialization(
        steps, V)
    cond = []
    for k in range(steps):
        print(t)
        t += dt
        _ = solve(sigma, sigma_n, dt, vector_field1, vector_field2, bc, phi, b, Wi, alpha, cond)

        save_solutions(sigma, sigma_11_solution_data, sigma_12_solution_data, sigma_21_solution_data,
                       sigma_22_solution_data, time_values_data, k, t)

        sigma.x.scatter_forward()
        sigma_n.x.array[:] = sigma.x.array
        t_n = t

    s1 = np.shape(time_values_data)[0]
    s2 = np.shape(sigma_11_solution_data)[1]
    tau = np.zeros((s1, 2, 2))
    for i in range(s1):
        B_prime = np.zeros((2, 2))
        for j in range(s2):
            A = np.array([[sigma_11_solution_data[i][j], sigma_12_solution_data[i][j]],
                          [sigma_21_solution_data[i][j], sigma_22_solution_data[i][j]]])

            trace_A = np.trace(A)
            factor = 1/(1 - trace_A / b)
            B = factor * A - np.identity(2)
            print(factor, A, B)

            B_prime += B

            # Store the result in P
        tau[i] = B_prime

    with open(np_path + 'sigma11.npy', 'wb') as f:
        np.save(f, np.array(sigma_11_solution_data))
    with open(np_path + 'sigma12.npy', 'wb') as f:
        np.save(f, np.array(sigma_12_solution_data))
    with open(np_path + 'sigma21.npy', 'wb') as f:
        np.save(f, np.array(sigma_21_solution_data))
    with open(np_path + 'sigma22.npy', 'wb') as f:
        np.save(f, np.array(sigma_22_solution_data))

    with open(np_path + 'tau.npy', 'wb') as f:
        np.save(f, tau)

    num_velocity_dofs = V.dofmap.index_map_bs * V.dofmap.index_map.size_global
    print(V.dofmap.index_map.size_global)
    print(V.dofmap.index_map_bs)
    fig = plt.figure(figsize=(25, 8))
    l1 = plt.plot(time_values_data, tau[:, 0, 0], label=r"FEniCSx  ({0:d} dofs)".format(num_velocity_dofs), linewidth=2)
    plt.title("tau_11 coefficient")
    plt.grid()
    plt.legend()
    plt.savefig(plot_path + "tau_11_comparison.png")

    fig = plt.figure(figsize=(25, 8))
    l1 = plt.plot(time_values_data, tau[:, 0, 1], label=r"FEniCSx  ({0:d} dofs)".format(num_velocity_dofs), linewidth=2)
    plt.title("tau_12 coefficient")
    plt.grid()
    plt.legend()
    plt.savefig(plot_path + "tau_12_comparison.png")

    fig = plt.figure(figsize=(25, 8))
    l1 = plt.plot(time_values_data, tau[:, 1, 0], label=r"FEniCSx  ({0:d} dofs)".format(num_velocity_dofs), linewidth=2)
    plt.title("tau_21 coefficient")
    plt.grid()
    plt.legend()
    plt.savefig(plot_path + "tau_21_comparison.png")

    fig = plt.figure(figsize=(25, 8))
    l1 = plt.plot(time_values_data, tau[:, 1, 1], label=r"FEniCSx  ({0:d} dofs)".format(num_velocity_dofs), linewidth=2)
    plt.title("tau_22 coefficient")
    plt.grid()
    plt.legend()
    plt.savefig(plot_path + "tau_22_comparison.png")


# pipeline()