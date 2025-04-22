from fenics import *
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import mshr
import os
import time
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300

plt.rcParams["font.family"] = "TeX Gyre Heros"
plt.rcParams["mathtext.fontset"] = "cm"
# ----------------------------------------------------------------------------------------------------------------------
# setup

# ----------------------------------------------------------------------------------------------------------------------
Nx = 32
experiment = 331
ud_experiment = 31  # "6_4b"
num_steps = 1
np_path = f"results/dolfin/OCP/experiments/{experiment}/"
timing_file = f"{np_path}timings.txt"
os.mkdir(np_path)
# ----------------------------------------------------------------------------------------------------------------------
with open("parameters.json", "r") as file:
    parameters = json.load(file)
    t0 = parameters["t0"]
    T = parameters["T"]
    h = parameters["dt"]
    alpha = parameters["alpha"]  # e-2
    LR = parameters["LR"]
    viscosity = parameters["viscosity"]
    K = parameters["buoy count"]
    mesh_boundary_x = parameters["mesh_boundary_x"]
    mesh_boundary_y = parameters["mesh_boundary_y"]
# ----------------------------------------------------------------------------------------------------------------------
# set_log_level(LogLevel.WARNING)
# rescaling cost
alpha = alpha * K
# ----------------------------------------------------------------------------------------------------------------------
# mesh = UnitSquareMesh(Nx, Nx)
left_x = 0.0
left_y = 2.0
right_x = 2.0
right_y = 2.0
mesh = RectangleMesh(Point(left_x, left_x), Point(right_x, right_y), Nx, Nx)
center_of_domain = np.array([1.0, 1.0])
Nx_t = 50
# rect1 = mshr.Rectangle(Point(0.0, 0.0), Point(2.0, 1.0))
# rect2 = mshr.Rectangle(Point(1.0, 0.0), Point(2.0, 2.0))
# rect2 = mshr.Rectangle(Point(1.0, 1.0), Point(2.0, 2.0))
# rect3 = mshr.Rectangle(Point(2.0, 0.0), Point(3.0, 0.8_8b))
# mesh = mshr.generate_mesh(rect1 + rect2, Nx_t)

# mesh_boundary = [[[0.0, 3.0], [0.0, 0.0]],    # t mesh lines    ___
#                  [[0.0, 0.0], [0.0, 1.0]],                      # |
#                  [[0.0, 1.0], [1.0, 1.0]],                      # _
#                  [[2.0, 3.0], [1.0, 1.0]],
#                  [[1.0, 2.0], [2.0, 2.0]],
#                  [[1.0, 1.0], [1.0, 2.0]],
#                  [[2.0, 2.0], [1.0, 2.0]],
#                  [[3.0, 3.0], [0.0, 1.0]]]
#
mesh_boundary = [[[0.0, 2.0], [0.0, 0.0]],
                 [[0.0, 0.0], [0.0, 2.0]],
                 [[0.0, 2.0], [2.0, 2.0]],
                 [[2.0, 2.0], [2.0, 0.0]]]


# mesh_boundary = [[[0.0, 2.0], [0.0, 0.0]],      # l shape
#                  [[0.0, 0.0], [0.0, 1.0]],
#                  [[0.0, 1.0], [1.0, 1.0]],
#                  [[1.0, 1.0], [1.0, 2.0]],
#                  [[1.0, 2.0], [2.0, 2.0]],
#                  [[2.0, 2.0], [2.0, 0.0]]]
plt.title(r"discretized domain $\Omega_h$")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plot(mesh)
plt.plot(mesh_boundary[0][0], mesh_boundary[0][1], color="blue")
plt.plot(mesh_boundary[1][0], mesh_boundary[1][1], color="orange", label=r"$\Gamma_1$")
plt.plot(mesh_boundary[2][0], mesh_boundary[2][1], color="blue", label=r"$\Gamma_2$")
plt.plot(mesh_boundary[3][0], mesh_boundary[3][1], color="orange")
# plt.plot(mesh_boundary[0][0], mesh_boundary[0][1], color="blue")
# plt.plot(mesh_boundary[1][0], mesh_boundary[1][1], color="orange", label=r"$\Gamma_1$")
# plt.plot(mesh_boundary[2][0], mesh_boundary[2][1], color="blue", label=r"$\Gamma_2$")
# plt.plot(mesh_boundary[3][0], mesh_boundary[3][1], color="blue")
# plt.plot(mesh_boundary[4][0], mesh_boundary[4][1], color="orange")
# plt.plot(mesh_boundary[5][0], mesh_boundary[5][1], color="blue")
plt.legend(loc="best", bbox_to_anchor=(1.02, 1))
plt.savefig(f"{np_path}mesh.png")
plt.clf()
# ----------------------------------------------------------------------------------------------------------------------

class Neumann(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (abs(x[0]) < DOLFIN_EPS or (abs(2.0 - x[0]) < DOLFIN_EPS))   # x[1] for L shape


NeumannBD = Neumann()

boundary_function = MeshFunction("size_t", mesh, mesh.geometric_dimension() - 1)
boundary_function.set_all(0)
NeumannBD.mark(boundary_function, 1)
# ----------------------------------------------------------------------------------------------------------------------
dx = Measure('dx', domain=mesh)
ds = Measure('ds', domain=mesh, subdomain_data=boundary_function)
# ----------------------------------------------------------------------------------------------------------------------
# f = Expression(('0.1', '0.0'), degree=2)
# f = Expression(("near(x[0], 0.0) ? f_left_x : (near(x[0], 1.0) ? f_right_x : 0.0)",
#                 "near(x[0], 0.0) ? f_left_y : (near(x[0], 1.0) ? f_right_y : 0.0)"),
#                degree=1,
#                f_left_x=0.1, f_left_y=0.0,   # Left boundary: (1,0)
#                f_right_x=-0.1, f_right_y=0.0)
# f_x = Expression("alpha * sin(pi*x[0]) * cos(pi*x[1])", alpha=1.0, degree=2)
# f_y = Expression("-alpha * cos(pi*x[0]) * sin(pi*x[1])", alpha=1.0, degree=2)
f_x = Expression("-cos(pi*x[0])*sin(pi*x[1])", degree=1)
f_y = Expression("sin(pi*x[0])*cos(pi*x[1])", degree=1)
f = as_vector([f_x, f_y])
df = Expression(('0.1', '0.1'), degree=2)
# ----------------------------------------------------------------------------------------------------------------------
P2 = VectorElement('CG', triangle, 2)
P1 = FiniteElement('CG', triangle, 1)
TH = MixedElement([P2, P1])
W = FunctionSpace(mesh, TH)
# ----------------------------------------------------------------------------------------------------------------------
ud_loaded = Function(W.sub(0).collapse())
with XDMFFile(f"results/dolfin/OCP/ud_construction/{ud_experiment}/paraview/velocity.xdmf") as infile:
    infile.read_checkpoint(ud_loaded, "u")

load_q = False
load_string = "243_f_LR_1.5"
if load_q:
    f = Function(W.sub(0).collapse())
    with XDMFFile(f"results/dolfin/OCP/experiments/{load_string}/q_backup/q.xdmf") as infile:
        infile.read_checkpoint(f, "f")


# ----------------------------------------------------------------------------------------------------------------------

def boundary(x, on_boundary):
    return on_boundary and (x[0] > DOLFIN_EPS and abs(2.0 - x[0]) > DOLFIN_EPS)  # x[1] for lshape


bcs = [DirichletBC(W.sub(0), (0, 0), boundary)]

# ----------------------------------------------------------------------------------------------------------------------
time_interval = np.linspace(t0, T, int(T / h))
u_d = np.zeros((K, int(T / h), mesh.geometric_dimension()))
x_d1 = np.zeros_like(time_interval)
x_d2 = np.zeros_like(time_interval)
ud_type = "custom_ud"

if ud_type == "circle":
    radius = 0.3
    middle_point_x = right_x / 2 - 0.25
    middle_point_y = right_y / 2
    ud1 = lambda t, al: right_x / 2 + radius * np.cos(2 * np.pi * t + al)
    ud2 = lambda t, al: right_x / 2 + radius * np.sin(2 * np.pi * t + al)
    ud1_back = lambda t: right_x / 2 - radius * np.sin(2 * np.pi * t)
    ud2_back = lambda t: right_x / 2 - radius * np.cos(2 * np.pi * t)

    x_d1 = middle_point_x + radius * np.cos(2 * np.pi * time_interval)
    x_d2 = middle_point_y + radius * np.sin(2 * np.pi * time_interval)
elif ud_type == "case2":
    ud1 = lambda t: 0.5 * (np.cos(np.pi * (t - 0.5)) - 1 - np.cos(np.pi))
    ud2 = lambda t: 0.0
    ud1_back = lambda t: - 0.5 * (np.cos(np.pi * (t - 0.5)) - 1 - np.cos(np.pi))
    ud2_back = lambda t: 0.0

    x_d1 = 0.5 + 0.25 * np.cos(2 * np.pi * time_interval)
    x_d2 = 0.5 + 0.25 * np.sin(2 * np.pi * time_interval)
elif ud_type == "diag_movement":
    # xsarr = [1.0 for _ in range(K)]
    xsarr = [0.5, 1.0, 1.5]
    # ysarr = np.linspace(1, 3, K)
    # ysarr = [0.5_2b for _ in range(K)]
    ysarr = [0.5, 0.5, 1.0]
    ud1 = lambda t: 0.5 * (np.cos(np.pi * (t - 0.5)) - 1 - np.cos(np.pi))
    ud2 = lambda t: 0.5 * (np.cos(np.pi * (t - 0.5)) - 1 - np.cos(np.pi))

    # x_d1 = [[x0, x0 + 1 / np.pi] for x0 in xsarr]
    # x_d2 = [[y0, y0 + 1 / np.pi] for y0 in ysarr]
    x_d1 = [[0.5, 0.5 + 1 / np.pi], [1., 1. + 1 / np.pi], [1.5, 1.5]]
    x_d2 = [[0.5, 0.5], [0.5, 0.5 + 1 / np.pi], [1.0, 1.0 + 1 / np.pi]]
elif ud_type == "arc":
    height = 0.1


    def ud1_movement(t):
        return 4 / (np.sqrt(4 + 64 * height ** 2 * (t - 0.5) ** 2))


    def ud2_movement(t):
        return (-8 * height * (t - 0.5)) / (np.sqrt(4 + 64 * height ** 2 * (t - 0.5) ** 2))


    ud1 = ud1_movement
    ud2 = ud2_movement
    x_d1 = [1.0 + 2 * time_interval for _ in range(K)]
    x_d2 = [i + height - 4 * height * (time_interval - 0.5) ** 2 for i in np.linspace(0.5, 1.5, K)]
    # x_d2 = [2.0 + height -4*height*(time_interval - 0.5_2b)**2 for _ in range(K)]
elif ud_type == "s-curve":
    umax = 1
    A = 0.1
    B = 0.1
    xsarr = [1.0 for _ in range(K)]
    ysarr = np.linspace(0.5, 1.5, K)


    def ud1_movement(t, xs):
        return ((xs + 2.0 - xs) + A * np.pi * np.cos(np.pi * t))  # *umax * np.sin(np.pi*t)**2


    def ud2_movement(t, ys):
        return ((ys + 0.25 - ys) + 2 * B * np.pi * np.cos(2 * np.pi * t))  # *umax * np.sin(np.pi*t)**2


    ud1 = ud1_movement
    ud2 = ud2_movement
    x_d1 = [i + (i + 2.0 - i) * time_interval + A * np.sin(np.pi * time_interval) for i in xsarr]
    x_d2 = [i + (i + 0.25 - i) * time_interval + B * np.sin(2 * np.pi * time_interval) for i in ysarr]

elif ud_type == "t-mesh":
    xsarr = [1.5, 1.5, 0.5, 0.5, 0.5]
    ysarr = [1.25, 0.5, 0.75, 0.5, 0.25]

    ud1 = [lambda t: 0.0, lambda t: 0.5 * (np.cos(np.pi * (t - 0.5)) - 1 - np.cos(np.pi))]
    ud2 = [lambda t: 0.5 * (np.cos(np.pi * (t - 0.5)) - 1 - np.cos(np.pi)), lambda t: 0.0]

    x_d1 = [[1.5, 1.5], [1.5, 1.5 + 1 / np.pi], [0.5, 0.5 + 1 / np.pi], [0.5, 0.5 + 1 / np.pi], [0.5, 0.5 + 1 / np.pi]]
    x_d2 = [[1.25, 1.25 + 1 / np.pi], [0.5, 0.5], [0.75, 0.75], [0.5, 0.5], [0.25, 0.25]]
elif ud_type == "custom_ud":
    with open(f"results/dolfin/OCP/ud_construction/{ud_experiment}/u_d_array.npy", "rb") as ud_reader:
        u_d = np.load(ud_reader)
    with open(f"results/dolfin/OCP/ud_construction/{ud_experiment}/x_0_array.npy", "rb") as ud_reader:
        temp = np.load(ud_reader)
        x_d1 = temp[:, :, 0]
        x_d2 = temp[:, :, 1]
        xsarr = temp[0:K + 1, 0, 0]
        ysarr = temp[0:K + 1, 0, 1]
else:  # horizontal movement
    xsarr = [1.0 for _ in range(K)]
    ysarr = np.linspace(1, 2, K)
    ud1 = lambda t: 0.5 * (np.cos(np.pi * (t - 0.5)) - 1 - np.cos(np.pi))
    ud2 = lambda t: 0.0
    ud1_back = lambda t: - 0.5 * (np.cos(np.pi * (t - 0.5)) - 1 - np.cos(np.pi))
    ud2_back = lambda t: 0.0
    x_d1 = [[x0, x0 + 1 / np.pi] for x0 in xsarr]
    x_d2 = [[y0, y0] for y0 in ysarr]

for k in range(K):
    if ud_type == "circle":
        if k == 0:
            al = 0
        elif k == 1:
            al = -np.pi / 2
        elif k == 2:
            al = - np.pi
        else:
            al = -1.5 * np.pi
    for i in range(len(time_interval)):
        # if i == 0 or i == 2:
        #     u_d[k, i, 0] = ud1(time_interval[i])
        #     u_d[k, i, 1] = ud2(time_interval[i])
        # else:
        #     u_d[k, i, 0] = ud1_back(time_interval[i])
        #     u_d[k, i, 1] = ud2_back(time_interval[i])
        if ud_type == "s-curve":
            u_d[k, i, 0] = ud1(time_interval[i], xsarr[k])  # , al)
            u_d[k, i, 1] = ud2(time_interval[i], ysarr[k])  # , al)
        elif ud_type == "t-mesh":
            if k != 1:
                u_d[k, i, 0] = ud1[0](time_interval[i])  # , al)
                u_d[k, i, 1] = ud2[0](time_interval[i])  # , al)

            else:
                u_d[k, i, 0] = ud1[1](time_interval[i])  # , al)
                u_d[k, i, 1] = ud2[1](time_interval[i])  # , al)
        elif ud_type == "diag_movement":
            if k == 0:
                u_d[k, i, 0] = ud1(time_interval[i])
                u_d[k, i, 1] = 0.0
            elif k == 1:
                u_d[k, i, 0] = ud1(time_interval[i])
                u_d[k, i, 1] = ud2(time_interval[i])
            elif k == 2:
                u_d[k, i, 0] = 0.0
                u_d[k, i, 1] = ud2(time_interval[i])
        elif ud_type != "custom_ud":
            u_d[k, i, 0] = ud1(time_interval[i])  # , al)
            u_d[k, i, 1] = ud2(time_interval[i])  # , al)

# plt.title(r"requested buoy movement $x$")
# plt.xlabel(r"$x_1$")
# plt.ylabel(r"$x_2$")
# ax = plt.gca()
# ax.set_aspect('equal', adjustable='box')
# for k in range(K):
#     plt.plot(x_d1[k], x_d2[k], label=r"$x$ of buoy " + f"{k + 1}")
# for line in mesh_boundary:
#     plt.plot(line[0], line[1], color="gray")
# plt.legend(loc="upper right")
# plt.savefig(f"{np_path}x_plot_buoys.png")


# ----------------------------------------------------------------------------------------------------------------------

def solve_primal_ode(wSol, buoy_mask):
    x = np.zeros((K, int(T / h), mesh.geometric_dimension()))
    x[:, 0, 0] = xsarr
    x[:, 0, 1] = ysarr

    u_values_array = np.zeros((K, int(T / h), mesh.geometric_dimension()))
    for b_iter in range(K):
        for k in range(int(T / h) - 1):
            print(f"primal ODE: ({b_iter}, {k})")
            point = np.array([x[b_iter, k, :][0].item(), x[b_iter, k, :][1].item()])
            try:
                u_values = wSol.sub(0)(point)
                x[b_iter, k + 1, :] = x[b_iter, k, :] + h * u_values
            except:
                print(b_iter)
                print("cant evaluate at")
                print(point)
                c = plot(wSol.sub(0), title="u_field")
                plt.colorbar(c)
                plt.savefig(f"{np_path}u_field.png")
                plt.clf()
                u_values = np.array([0.0, 0.0])
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


def solve_adjoint_ode(wSol, grad_u, x, buoy_mask, u_values_array):
    mu = np.zeros((K, int(T / h), mesh.geometric_dimension()))
    for b_iter in range(K):
        if buoy_mask[b_iter]:
            continue
        N = len(time_interval[1:])
        for k in range(N - 1, -1, -1):
            #print(f"adjoint ODE: ({b_iter}, {k})")
            point = np.array([x[b_iter, k + 1, :][0].item(), x[b_iter, k + 1, :][1].item()])
            try:
                # u_values = wSol.sub(0)(point)
                grad_u_values = grad_u(point)
                grad_u_matr = np.array([[grad_u_values[0].item(), grad_u_values[1].item()],
                                        [grad_u_values[2].item(), grad_u_values[3].item()]])
            except:
                print(b_iter)
                print("cant evaluate at")
                print(point)
                c = plot(wSol.sub(0), title="u_field")
                plt.colorbar(c)
                plt.savefig(f"{np_path}u_field.png")
                plt.clf()
                print("setting buoy to middle point")
                # u_values = np.array([0.0, 0.0])
                # grad_u_matr = np.array([[0.0, 0.0],
                #                         [0.0, 0.0]])
            mu[b_iter, k, :] = mu[b_iter, k + 1, :] - h * grad_u_matr.T @ (
                (u_values_array[b_iter, k+1, :] - u_d[b_iter, k + 1, :])/1 - mu[b_iter, k + 1, :])
    return mu


# ----------------------------------------------------------------------------------------------------------------------
# function space for gradient u
V_vec = TensorFunctionSpace(mesh, "Lagrange", 1)


# ----------------------------------------------------------------------------------------------------------------------


def J(u__, f_):
    # partA = assemble(0.5_2b * inner(u - u_d, u - u_d) * dx)
    partA = (0.5/1) * np.sum(np.sum(h * (np.linalg.norm(u__ - u_d, axis=2) ** 2), axis=1))
    partB = assemble(alpha * 0.5 * inner(f_, f_) * ds(int(1)))
    return partA + partB


# ----------------------------------------------------------------------------------------------------------------------
J_array = []
x_array = []
divs_u = []


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
os.mkdir(np_path + "buoy_movements")
os.mkdir(np_path + "flow_fields")
os.mkdir(np_path + "paraview")
os.mkdir(np_path + "paraview/checkpoint")
os.mkdir(np_path + "checkpoints")
os.mkdir(np_path + "q_backup")
os.mkdir(np_path + "buoy_movements/frames")
# ----------------------------------------------------------------------------------------------------------------------
checkpoints = False
if checkpoints:
    f = Function(W.sub(0).collapse())
    with XDMFFile(f"results/dolfin/OCP/experiments/{experiment - 1}/checkpoints/q.xdmf") as infile:
        infile.read_checkpoint(f, "f")
# ----------------------------------------------------------------------------------------------------------------------
ns_iterations = []
tau = 0.5
tau_inc = 2.0
c = 1e-4
LR_MIN = 1e-6
LR_MAX = 5
LR = LR_MAX
conv_crit = 1e-3
# optimization loop
outer_timing_array = []
inner_timing_array = []
inner_iterations_array = []
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

    # A, b = assemble_system(aAdj, FAdj, bcs)
    # u_values = np.zeros((K, int(T / h), mesh.geometric_dimension()))
    for buoy, points in enumerate(x):
        if buoy_mask[buoy]:
            continue
        for k, point in enumerate(points):
            # u_values[buoy, k, :] = w.sub(0)(point)
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

    end_outer_loop = time.time()
    duration_outer_loop = end_outer_loop - start_outer_loop
    # ----------------------------------------------------------------------------------------------------------------------
    # gradient check
    # gradj = assemble(inner(alpha * f - zSol, df) * ds(int(1)))
    if i == 0:
        J0 = J(u_values_array, f)
        # print(J0)
        # grad_test(a, v, w, J0, gradj, i, buoy_mask)

    # ----------------------------------------------------------------------------------------------------------------------
    # line search for gradient update
    start_inner_loop = time.time()
    inner_iterations = 0
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
            # LR = min(tau_inc * LR, LR_MAX)
            break
        LR = max(tau * LR, LR_MIN)  # max(tau * LR, LR_MIN)

    end_inner_loop = time.time()
    duration_inner_loop = end_inner_loop - start_inner_loop
    outer_timing_array.append(duration_outer_loop)
    inner_timing_array.append(duration_inner_loop)
    inner_iterations_array.append(inner_iterations)

    # control update
    f = f - LR * (alpha * f - zSol)

    # collecting data
    J_array.append(J(u_values_array, f))
    divs_u.append(sqrt(assemble(div(u) * div(u) * dx)))

    plt.clf()
    uplot = plot(w.sub(0), title=f"u_{i}_field")
    plt.colorbar(uplot)
    plt.savefig(f"{np_path}flow_fields/u_{i}_field.png")
    plt.clf()

    with XDMFFile(f"{np_path}checkpoints/q.xdmf") as outfile:
        outfile.write_checkpoint(project(f, W.sub(0).collapse()), "f", 0, append=True)

    # condition check
    if i > 5 and abs((J_array[i] - J_array[i - 1])) < conv_crit:  # / J_array[i - 1]) < 1e-4:
        print("cost small enough")
        break
    elif sum(buoy_mask) > 10:
        print("more than ten buoys out of domain .. exiting")
        break

# ----------------------------------------------------------------------------------------------------------------------
# ns_iters = 0
# for i in ns_iterations:
#     ns_iters += i
# print(f"average NS: iterations: {ns_iters/len(ns_iterations)}")
# ----------------------------------------------------------------------------------------------------------------------
with open(timing_file, "w") as time_writer:

    for k, i in enumerate(inner_iterations_array):
        time_writer.write(f"Iteration {k}:\n")
        time_writer.write(f"  outer loop time: {outer_timing_array[k]:.6f} seconds\n")
        time_writer.write(f"  inner loop time: {inner_timing_array[k]:.6f} seconds\n")
        time_writer.write(f"  inner loop iterations: {i}\n")
        time_writer.write("-" * 40 + "\n")
# ----------------------------------------------------------------------------------------------------------------------
with XDMFFile(f"{np_path}q_backup/q.xdmf") as outfile:
    outfile.write_checkpoint(project(f, W.sub(0).collapse()), "f", 0, append=True)

# ||u_m - u|| output
u_bar_loaded = Function(W.sub(0).collapse())
with XDMFFile(f"results/dolfin/OCP/experiments/{165}/paraview/checkpoint/u.xdmf") as infile:
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
    text_file.write("mesh t shape resolution: %s \n" % Nx_t)
    text_file.write("ud type: %s \n" % ud_type)
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
# L2_norm = sqrt(assemble((u - ud_loaded) ** 2 * dx))
# H1_norm = sqrt(assemble(((u - ud_loaded) ** 2 + inner(grad((u - ud_loaded)), grad((u - ud_loaded)))) * dx))
# with open(np_path + "norms.txt", "w") as text_file:
#     text_file.write("L2: %s \n" % L2_norm)
#     text_file.write("H1: %s \n" % H1_norm)
# ----------------------------------------------------------------------------------------------------------------------
# plotting
print("plotting")
plt.clf()
plt.xlabel(r"Iteration")
plt.ylabel(r"Cost")
plt.title(r"Reduced cost $j(q)$")
plt.plot(J_array, color="black")
plt.savefig(f"{np_path}J.png")
plt.clf()

with open(f"{np_path}J_array.npy", "wb") as np_writer:
    np.save(np_writer, J_array)


def generate_dotted_style(k):
    base = k + 1  # Increase the base to vary patterns
    return (0, (base, base // 2))


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

c = plot(u, title=r"Velocity field $u$")
plt.colorbar(c)
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.savefig(f"{np_path}u_field.png")
plt.clf()

ux, px = w.split()
xdmf_file = XDMFFile(f"{np_path}paraview/velocity.xdmf")
xdmf_file.write_checkpoint(ux, "u", 0)
xdmf_filep = XDMFFile(f"{np_path}paraview/pressure.xdmf")
xdmf_filep.write_checkpoint(px, "p", 0)

xdmf_file = XDMFFile(f"{np_path}paraview/checkpoint/u.xdmf")
xdmf_file.write_checkpoint(ux, "u", 0)
xdmf_filep = XDMFFile(f"{np_path}paraview/checkpoint/p.xdmf")
xdmf_filep.write_checkpoint(px, "p", 0)
