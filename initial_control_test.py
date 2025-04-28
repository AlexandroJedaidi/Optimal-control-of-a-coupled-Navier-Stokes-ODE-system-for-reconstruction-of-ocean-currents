from fenics import *
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import mshr
import os
import time
import matplotlib as mpl
# ----------------------------------------------------------------------------------------------------------------------
# plot resolution
mpl.rcParams['figure.dpi'] = 300
# here we load fonts for plots, comment it out, if it throws errors
plt.rcParams["font.family"] = "TeX Gyre Heros"
# here we load fonts for plots, comment it out, if it throws errors
plt.rcParams["mathtext.fontset"] = "cm"
# ----------------------------------------------------------------------------------------------------------------------
# change these parameters
experiment = 1    # set to experiment number, all data can be found under this experiment number
ud_experiment = "6_buoys"   # from this file the u_d and x_d data gets loaded, see reference_runs/
num_steps = 50   # max number of Gradient descent iterations

np_path = f"results/dolfin/OCP/experiments/{experiment}/"  # destination path where everything gets stored

unit_square_resolution = 32     # mesh resolution of unit square

grad_check = False  # Set to false, if gradients shouldn't be checked

# setup initial q in this pipeline
case = 0    # choose initial control
if case == 0:
    q_0_1 = Expression("-cos(pi*x[0])*sin(pi*x[1])", degree=1)
    q_0_2 = Expression("sin(pi*x[0])*cos(pi*x[1])", degree=1)
elif case == 1:
    q_0_1 = Expression("0.0", degree=1)
    q_0_2 = Expression("0.0", degree=1)
elif case == 2:
    q_0_1 = Expression("sin(pi*x[0])*cos(pi*x[1])", degree=1)
    q_0_2 = Expression("-cos(pi*x[0])*sin(pi*x[1])", degree=1)
else:
    q_0_1 = Expression("0.1", degree=1)
    q_0_2 = Expression("0.1", degree=1)


# line search setup
use_line_search = False     # Set to False if Line search should be turned off
tau = 0.5
c = 1e-4  # Armijo Condition threshold
LR_MIN = 1e-6  # min Learning rate
LR_MAX = 5  # max learning rate
LR = LR_MAX  # learning rate is set to max learning rate (to utilize line search), set to own LR, if

# |j(q) - j(q_old)| < conv_crit
conv_crit = 1e-3
# ----------------------------------------------------------------------------------------------------------------------
# for debugging
load_q = False  # load q from another run
load_string = "243_f_LR_1.5"  # set this to experiment name where q should be loaded from
checkpoints = False  # restarts the run with checkpoint control
# ----------------------------------------------------------------------------------------------------------------------
# create a folder where results gets saved to
timing_file = f"{np_path}timings.txt"
directory = os.path.dirname(np_path)
if not os.path.exists(directory):
    os.makedirs(directory)
    os.mkdir(np_path + "buoy_movements")
    os.mkdir(np_path + "flow_fields")
    os.mkdir(np_path + "paraview")
    os.mkdir(np_path + "paraview/checkpoint")
    os.mkdir(np_path + "checkpoints")
    os.mkdir(np_path + "q_backup")
    os.mkdir(np_path + "buoy_movements/frames")
# ----------------------------------------------------------------------------------------------------------------------
with open("parameters.json", "r") as file:
    parameters = json.load(file)
    t0 = parameters["t0"]
    T = parameters["T"]
    h = parameters["dt"]
    alpha = parameters["alpha"]
    # LR = parameters["LR"]
    viscosity = parameters["viscosity"]
    K = parameters["buoy count"]
# ----------------------------------------------------------------------------------------------------------------------
# rescaling cost
alpha = alpha * K
# ----------------------------------------------------------------------------------------------------------------------
# mesh init
Nx = unit_square_resolution
left_x = 0.0
left_y = 2.0
right_x = 2.0
right_y = 2.0
center_of_domain = np.array([1.0, 1.0])
mesh = RectangleMesh(Point(left_x, left_x), Point(right_x, right_y), Nx, Nx)
mesh_boundary = [[[0.0, 2.0], [0.0, 0.0]],
                 [[0.0, 0.0], [0.0, 2.0]],
                 [[0.0, 2.0], [2.0, 2.0]],
                 [[2.0, 2.0], [2.0, 0.0]]]
# ----------------------------------------------------------------------------------------------------------------------
# Function space
P2 = VectorElement('CG', triangle, 2)
P1 = FiniteElement('CG', triangle, 1)
TH = MixedElement([P2, P1])
W = FunctionSpace(mesh, TH)

# function space for grad(u)
V_vec = TensorFunctionSpace(mesh, "Lagrange", 1)
# ----------------------------------------------------------------------------------------------------------------------
# Boundary conditions
class Neumann(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (abs(x[0]) < DOLFIN_EPS or (abs(2.0 - x[0]) < DOLFIN_EPS))


NeumannBD = Neumann()

boundary_function = MeshFunction("size_t", mesh, mesh.geometric_dimension() - 1)
boundary_function.set_all(0)
NeumannBD.mark(boundary_function, 1)

def boundary(x, on_boundary):
    return on_boundary and (x[0] > DOLFIN_EPS and abs(2.0 - x[0]) > DOLFIN_EPS)


bcs = [DirichletBC(W.sub(0), (0, 0), boundary)]
# ----------------------------------------------------------------------------------------------------------------------
# Integral measure init. ds is for integrals on \Gamma_1
dx = Measure('dx', domain=mesh)
ds = Measure('ds', domain=mesh, subdomain_data=boundary_function)
# ----------------------------------------------------------------------------------------------------------------------
# q_0 initialization
f_x = q_0_1#Expression("-cos(pi*x[0])*sin(pi*x[1])", degree=1)
f_y = q_0_2#Expression("sin(pi*x[0])*cos(pi*x[1])", degree=1)
f = as_vector([f_x, f_y])

# used for directional derivatives
df = Expression(('0.1', '0.1'), degree=2)
# ----------------------------------------------------------------------------------------------------------------------
# load q from other run
if load_q:
    f = Function(W.sub(0).collapse())
    with XDMFFile(f"results/dolfin/OCP/experiments/{load_string}/q_backup/q.xdmf") as infile:
        infile.read_checkpoint(f, "f")
# ----------------------------------------------------------------------------------------------------------------------
# load q from checkpoint when high computation times
if checkpoints:
    f = Function(W.sub(0).collapse())
    with XDMFFile(f"results/dolfin/OCP/experiments/{experiment}/checkpoints/q.xdmf") as infile:
        infile.read_checkpoint(f, "f")
# ----------------------------------------------------------------------------------------------------------------------
# prepare arrays for ODE solving and plotting
time_interval = np.linspace(t0, T, int(T / h))
u_d = np.zeros((K, int(T / h), mesh.geometric_dimension()))
x_d1 = np.zeros_like(time_interval)
x_d2 = np.zeros_like(time_interval)

with open(f"reference_runs/{ud_experiment}/u_d_array.npy", "rb") as ud_reader:
    u_d = np.load(ud_reader)
with open(f"reference_runs/{ud_experiment}/x_0_array.npy", "rb") as ud_reader:
    temp = np.load(ud_reader)
    x_d1 = temp[:, :, 0]
    x_d2 = temp[:, :, 1]
    xsarr = temp[0:K + 1, 0, 0]
    ysarr = temp[0:K + 1, 0, 1]

# ----------------------------------------------------------------------------------------------------------------------
# helper function to solve primal ODE
def solve_primal_ode(wSol, buoy_mask):
    x = np.zeros((K, int(T / h), mesh.geometric_dimension()))
    x[:, 0, 0] = xsarr
    x[:, 0, 1] = ysarr

    u_values_array = np.zeros((K, int(T / h), mesh.geometric_dimension()))
    for b_iter in range(K):
        for k in range(int(T / h) - 1):
            point = np.array([x[b_iter, k, :][0].item(), x[b_iter, k, :][1].item()])
            try:
                u_values = wSol.sub(0)(point)
                x[b_iter, k + 1, :] = x[b_iter, k, :] + h * u_values
            except:
                print(b_iter)
                print("cant evaluate at")
                print(point)
                print("setting x to center of domain")
                x[b_iter, :, 0] = center_of_domain[0] * np.ones_like(x[b_iter, :, 0])
                x[b_iter, :, 1] = center_of_domain[1] * np.ones_like(x[b_iter, :, 1])
                buoy_mask[b_iter] = True
                break
            u_values_array[b_iter, k, :] = u_values
        try:
            u_values_array[b_iter, k + 1, :] = wSol.sub(0)(
                np.array([x[b_iter, k + 1, :][0].item(), x[b_iter, k + 1, :][1].item()]))
        except:
            print("buoy ran out, setting velocity to zero and x to middle of domain")
            u_values_array[b_iter, k + 1, :] = np.array([0.0, 0.0])
            x[b_iter, k + 1, :] = center_of_domain
    return x, u_values_array

# helper function to solve adjoint ODE
def solve_adjoint_ode(wSol, grad_u, x, buoy_mask, u_values_array):
    mu = np.zeros((K, int(T / h), mesh.geometric_dimension()))
    for b_iter in range(K):
        if buoy_mask[b_iter]:
            continue
        N = len(time_interval[1:])
        for k in range(N - 1, -1, -1):
            point = np.array([x[b_iter, k + 1, :][0].item(), x[b_iter, k + 1, :][1].item()])
            try:
                grad_u_values = grad_u(point)
                grad_u_matr = np.array([[grad_u_values[0].item(), grad_u_values[1].item()],
                                        [grad_u_values[2].item(), grad_u_values[3].item()]])
            except:
                print(b_iter)
                print("cant evaluate at")
                print(point)
                c = plot(wSol.sub(0), title="u_field")
            mu[b_iter, k, :] = mu[b_iter, k + 1, :] - h * grad_u_matr.T @ (
                (u_values_array[b_iter, k+1, :] - u_d[b_iter, k + 1, :])/1 - mu[b_iter, k + 1, :])
    return mu


# ----------------------------------------------------------------------------------------------------------------------
# helper function for cost function
def J(u__, f_):
    # partA = assemble(0.5_2b * inner(u - u_d, u - u_d) * dx)
    partA = (0.5/1) * np.sum(np.sum(h * (np.linalg.norm(u__ - u_d, axis=2) ** 2), axis=1))
    partB = assemble(alpha * 0.5 * inner(f_, f_) * ds(int(1)))
    return partA + partB



# ----------------------------------------------------------------------------------------------------------------------

def grad_test(a, v, w, J0, gradj, iter, buoy_mask):
    with open(np_path + f"grad_J_error_{iter}.txt", "w") as text_file:
        text_file.write("reduced Gradient j \t \t approximated gradient J \t Error \t \t \t h_i \n")
        for k in range(1, 9):
            h_ = 10 ** (-k)
            F = a - inner(f + h_ * df, v) * ds(int(1))
            solve(F == 0, w, bcs)
            _, u_values_array = solve_primal_ode(w, buoy_mask)
            gradapprox = (J(u_values_array, f + h_ * df) - J0) / h_
            text_file.write(f" {gradj} \t {gradapprox} \t {abs(gradapprox - gradj)} \t {h_} \n")
        text_file.close()
    with open(np_path + f"grad_J_error_centered_{iter}.txt", "w") as text_file:
        text_file.write("reduced Gradient j \t \t approximated gradient J \t Error \t \t \t h_i \n")
        for k in range(1, 9):
            h_ = 10 ** (-k)
            F = a - inner(f + h_ * df, v) * ds(int(1))
            solve(F == 0, w, bcs)
            _, u_values_array = solve_primal_ode(w, buoy_mask)
            jEvalRight = J(u_values_array, f + h_ * df)

            F = a - inner(f - h_ * df, v) * ds(int(1))
            solve(F == 0, w, bcs)
            _, u_values_array = solve_primal_ode(w, buoy_mask)
            jEvalLeft = J(u_values_array, f - h_ * df)
            gradapprox = (jEvalRight - jEvalLeft) / (2 * h_)

            text_file.write(f" {gradj} \t {gradapprox} \t {abs(gradj - gradapprox)} \t {h_} \n")
        text_file.close()

# ----------------------------------------------------------------------------------------------------------------------
# collecting data in arrays for plotting
J_array = []
x_array = []
divs_u = []
outer_timing_array = []
inner_timing_array = []
inner_iterations_array = []
# ----------------------------------------------------------------------------------------------------------------------
# optimization loop
for i in range(num_steps):
    buoy_mask = np.zeros(K)
    # ----------------------------------------------------------------------------------------------------------------------
    start_outer_loop = time.time()
    # solving primal NS
    w = Function(W)
    w_test = TestFunction(W)

    u, p = split(w)
    v, q = split(w_test)
    n = FacetNormal(mesh)
    a = (viscosity * inner(grad(u), grad(v)) + inner(dot(grad(u), u), v) + div(u) * q + div(v) * p) * dx - 0.5 * (
        dot(dot(u, n) * u, v)) * ds(int(1))
    F = a - inner(f, v) * ds(int(1))

    solve(F == 0, w, bcs)
    # ----------------------------------------------------------------------------------------------------------------------
    # solving primal and adjoint ODE
    grad_u = grad(w.sub(0))
    grad_u_proj = project(grad_u, V_vec)
    x, u_values_array = solve_primal_ode(w, buoy_mask)
    mu = solve_adjoint_ode(w, grad_u_proj, x, buoy_mask, u_values_array)

    x_array.append(x)
    # ----------------------------------------------------------------------------------------------------------------------
    # solving adjoint PDE
    zrSol = Function(W)
    zSol, rSol = split(zrSol)

    w_ad = TrialFunction(W)
    u_ad, p_ad = split(w_ad)
    vq_ad = TestFunction(W)
    v_ad, q_ad = split(vq_ad)

    aAdj = (inner(grad(u_ad), grad(v_ad)) + inner(grad(w.sub(0)) * v_ad, u_ad) + inner(grad(v) * w.sub(0),
                                                                                       u_ad) + div(
        u_ad) * q_ad + div(v_ad) * p_ad) * dx - 0.5 * (
                   (dot(dot(w.sub(0), n) * v_ad, u_ad)) + (dot(dot(v_ad, n) * w.sub(0), u_ad))) * ds(int(1))
    FAdj = inner(Constant((0.0, 0.0)), v_ad) * dx

    A = assemble(aAdj)
    b = assemble(FAdj)

    for buoy, points in enumerate(x):
        if buoy_mask[buoy]:
            continue
        for k, point in enumerate(points):
            try:
                u_x = w.sub(0)(point)
            except:
                u_x = np.array([0.0, 0.0])
                point = center_of_domain
            gamma = h * ((u_d[buoy, k, :] - u_x)/1 + mu[buoy, k, :])
            delta0 = PointSource(W.sub(0).sub(0), Point(point), gamma[0])
            delta1 = PointSource(W.sub(0).sub(1), Point(point), gamma[1])
            delta0.apply(b)
            delta1.apply(b)

    bcs[0].apply(A)
    bcs[0].apply(b)

    solve(A, zrSol.vector(), b)
    # ----------------------------------------------------------------------------------------------------------------------
    # save timings
    end_outer_loop = time.time()
    duration_outer_loop = end_outer_loop - start_outer_loop
    # ----------------------------------------------------------------------------------------------------------------------
    # gradient check
    if i == 0 and grad_check:
        J0 = J(u_values_array, f)
        gradj = assemble(inner(alpha * f - zSol, df) * ds(int(1)))
        grad_test(a, v, w, J0, gradj, i, buoy_mask)
    # ----------------------------------------------------------------------------------------------------------------------
    # line search for gradient update
    start_inner_loop = time.time()
    inner_iterations = 0
    if use_line_search:
        df = - (alpha * f - zSol)
        gradj = assemble(inner(alpha * f - zSol, df) * ds(int(1)))
        cond = - c * gradj
        while True:
            print("line search at " + str(LR))
            inner_iterations += 1
            buoy_mask_LS = np.zeros(K)
            J_old = J(u_values_array, f)
            w_ls = Function(W)
            w_test_ls = TestFunction(W)

            u_ls, p_ls = split(w_ls)
            v_ls, q_ls = split(w_test_ls)
            f_ls = f + LR * df
            a_ls = (viscosity * inner(grad(u_ls), grad(v_ls)) + inner(dot(grad(u_ls), u_ls), v_ls) + div(
                u_ls) * q_ls + div(
                v_ls) * p_ls) * dx - 0.5 * (dot(dot(u_ls, n) * u_ls, v_ls)) * ds(int(1))
            F_ls = a_ls - inner(f_ls, v_ls) * ds(int(1))

            solve(F_ls == 0, w_ls, bcs)
            # ----------------------------------------------------------------------------------------------------------------------

            grad_u_ls = grad(w_ls.sub(0))
            grad_u_proj_ls = project(grad_u_ls, V_vec)
            x_ls, u_values_array_ls = solve_primal_ode(w_ls, buoy_mask_LS)
            J_new = J(u_values_array_ls, f_ls)
            if J_old - J_new >= LR * cond:
                break
            LR = max(tau * LR, LR_MIN)  # max(tau * LR, LR_MIN)
    # ------------------------------------------------------------------------------------------------------------------
    # save timings
    end_inner_loop = time.time()
    duration_inner_loop = end_inner_loop - start_inner_loop
    outer_timing_array.append(duration_outer_loop)
    inner_timing_array.append(duration_inner_loop)
    inner_iterations_array.append(inner_iterations)
    # ------------------------------------------------------------------------------------------------------------------
    # control update
    f = f - LR * (alpha * f - zSol)
    # ------------------------------------------------------------------------------------------------------------------
    # collecting data
    J_array.append(J(u_values_array, f))
    divs_u.append(sqrt(assemble(div(u) * div(u) * dx)))
    # ------------------------------------------------------------------------------------------------------------------
    # collecting flow profiles in every iteration
    plt.clf()
    uplot = plot(w.sub(0), title=f"u_{i}_field")
    plt.colorbar(uplot)
    plt.savefig(f"{np_path}flow_fields/u_{i}_field.png")
    plt.clf()
    # ------------------------------------------------------------------------------------------------------------------
    # save control checkpoint
    with XDMFFile(f"{np_path}checkpoints/q.xdmf") as outfile:
        outfile.write_checkpoint(project(f, W.sub(0).collapse()), "f", 0, append=True)
    # ------------------------------------------------------------------------------------------------------------------
    # convergence condition check
    if i > 5 and abs((J_array[i] - J_array[i - 1])) < conv_crit:  # / J_array[i - 1]) < 1e-4:
        print("cost small enough")
        break
    # if more than half of all buoys flow out of the domain, we exit, we can assume not enough measurements to converge
    elif sum(buoy_mask) > 10:
        print("more than ten buoys out of domain .. exiting")
        break

# ----------------------------------------------------------------------------------------------------------------------
# visualization and saving data
# plotting mesh
plt.title(r"discretized domain $\Omega_h$")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plot(mesh)
plt.plot(mesh_boundary[0][0], mesh_boundary[0][1], color="blue")
plt.plot(mesh_boundary[1][0], mesh_boundary[1][1], color="orange", label=r"$\Gamma_1$")
plt.plot(mesh_boundary[2][0], mesh_boundary[2][1], color="blue", label=r"$\Gamma_2$")
plt.plot(mesh_boundary[3][0], mesh_boundary[3][1], color="orange")
plt.legend(loc="best", bbox_to_anchor=(1.02, 1))
plt.savefig(f"{np_path}mesh.png")
plt.clf()
# ----------------------------------------------------------------------------------------------------------------------
# saving calculation times
with open(timing_file, "w") as time_writer:
    for k, i in enumerate(inner_iterations_array):
        time_writer.write(f"Iteration {k}:\n")
        time_writer.write(f"  outer loop time: {outer_timing_array[k]:.6f} seconds\n")
        time_writer.write(f"  inner loop time: {inner_timing_array[k]:.6f} seconds\n")
        time_writer.write(f"  inner loop iterations: {i}\n")
        time_writer.write("-" * 40 + "\n")
# ----------------------------------------------------------------------------------------------------------------------
# save control to continue in another run
with XDMFFile(f"{np_path}q_backup/q.xdmf") as outfile:
    outfile.write_checkpoint(project(f, W.sub(0).collapse()), "f", 0, append=True)
# ----------------------------------------------------------------------------------------------------------------------
# ||u_m - u|| output
u_bar_loaded = Function(W.sub(0).collapse())
with XDMFFile(f"reference_runs/u_bar_chapter_6.3.3/paraview/checkpoint/u.xdmf") as infile:
    infile.read_checkpoint(u_bar_loaded, "u", 0)

l2_norm = sqrt(assemble(inner(u - u_bar_loaded, u - u_bar_loaded) * dx))
H1_norm = sqrt(assemble(inner(u - u_bar_loaded, u - u_bar_loaded) * dx) + assemble(
    inner(grad(u - u_bar_loaded), grad(u - u_bar_loaded)) * dx))

with open(np_path + "norm_table.txt", "w") as text_file:
    text_file.write("l2 \t \t \t h1  \n")
    text_file.write(f" {l2_norm} \t {H1_norm} \n")
# ----------------------------------------------------------------------------------------------------------------------
# divergence output
with open(np_path + "u_divergence.txt", "w") as text_file:
    for i, div_u_val in enumerate(divs_u):
        text_file.write("div(u) \t \t \t i  \n")
        text_file.write(f" {div_u_val} \t {i} \n")
# ----------------------------------------------------------------------------------------------------------------------
# parameter output
with open(np_path + "variables.txt", "w") as text_file:
    text_file.write("mesh resolution: %s \n" % Nx)
    text_file.write("t0: %s \n" % t0)
    text_file.write("T: %s \n" % T)
    text_file.write("dt: %s \n" % h)
    text_file.write("viscosity: %s \n" % viscosity)
    text_file.write("buoy count: %s \n" % K)
    text_file.write("LR: %s \n" % LR)
    text_file.write("LR_MAX: %s \n" % LR_MAX)
    text_file.write("LR_MIN: %s \n" % LR_MIN)
    text_file.write("conv. crit.: %s \n" % conv_crit)
    text_file.write("gradient descent steps: %s \n" % num_steps)
# ----------------------------------------------------------------------------------------------------------------------
# save cost array for joined cost plots
with open(f"{np_path}J_array.npy", "wb") as np_writer:
    np.save(np_writer, J_array)
# ----------------------------------------------------------------------------------------------------------------------
# Cost plot
print("plotting")
plt.clf()
plt.xlabel(r"Iteration")
plt.ylabel(r"Cost")
plt.title(r"Reduced cost $j(q)$")
plt.plot(J_array, color="black")
plt.savefig(f"{np_path}J.png")
plt.clf()
# ----------------------------------------------------------------------------------------------------------------------



def generate_dotted_style(k):   # use dynamic dotted plots
    base = k + 1  # Increase the base to vary patterns
    return (0, (base, base // 2))

# plot buoy movement comparison
for k, x_ in enumerate(x_array):
    color = plt.cm.rainbow(np.linspace(0, 1, K))
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.title(r"Buoy movement result")
    for i, x_buoy in enumerate(x_):
        plt.scatter(xsarr[i], ysarr[i], color="red", zorder=5)
        plt.text(xsarr[i], ysarr[i] + 0.1, rf"$x_{i + 1}(0)$", ha='center', va='center')
        linestyle = generate_dotted_style(i + 1)
        x_coord = x_buoy[:, 0]
        y_coord = x_buoy[:, 1]
        # plt.xlim(0.0, 3)
        # plt.ylim(0.0, 3)
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.plot(x_d1[i], x_d2[i], label=r"$x_d$" if i == 0 else "", color="black", alpha=0.5)
        plt.plot(x_coord, y_coord, label=rf"$x_{i + 1}$", color="b", linestyle=linestyle)

    for line in mesh_boundary:
        plt.plot(line[0], line[1], color="gray")
    plt.legend(loc="best", bbox_to_anchor=(1.02, 1))
    plt.savefig(f"{np_path}buoy_movements/frames/buoy_movement_{k}.png")
    plt.clf()
# ----------------------------------------------------------------------------------------------------------------------
# plot velocity comparison
for k in range(K):
    linestyle = generate_dotted_style(k + 1)
    plt.title(rf"Velocity comparison for buoy k={k + 1}")
    plt.xlabel("Time")
    plt.ylabel("Velocity")
    plt.plot(time_interval, u_d[k, :, 0], label=rf"$u_{{d,{1}}}$", color="black", alpha=0.8)
    plt.plot(time_interval, u_d[k, :, 1], label=rf"$u_{{d,{2}}}$", color="black", alpha=0.8)
    plt.plot(time_interval, u_values_array[k, :, 0], label=r"$u_{1}$", linestyle=linestyle, color="b")
    plt.plot(time_interval, u_values_array[k, :, 1], label=r"$u_{2}$", linestyle=linestyle, color="b")

    plt.legend(loc="best")
    plt.savefig(f"{np_path}ud_plot_buoy_{k}.png")
    plt.clf()
# ----------------------------------------------------------------------------------------------------------------------
# plot velocity field
c = plot(u, title=r"Velocity field $u$")
plt.colorbar(c)
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.savefig(f"{np_path}u_field.png")
plt.clf()
# ----------------------------------------------------------------------------------------------------------------------
# save u and p for paraview analysis
ux, px = w.split()
xdmf_file = XDMFFile(f"{np_path}paraview/velocity.xdmf")
xdmf_file.write_checkpoint(ux, "u", 0)
xdmf_filep = XDMFFile(f"{np_path}paraview/pressure.xdmf")
xdmf_filep.write_checkpoint(px, "p", 0)
# ----------------------------------------------------------------------------------------------------------------------
# save u and p as checkpoints for reruns
xdmf_file = XDMFFile(f"{np_path}paraview/checkpoint/u.xdmf")
xdmf_file.write_checkpoint(ux, "u", 0)
xdmf_filep = XDMFFile(f"{np_path}paraview/checkpoint/p.xdmf")
xdmf_filep.write_checkpoint(px, "p", 0)
# ----------------------------------------------------------------------------------------------------------------------
