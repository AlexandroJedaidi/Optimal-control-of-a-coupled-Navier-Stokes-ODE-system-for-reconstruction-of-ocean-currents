from fenics import *
import json
import numpy as np
import os
import matplotlib.pyplot as plt

Nx = 32
experiment = 56
num_steps = 150
np_path = f"results/dolfin/OCP/experiments/{experiment}/"
#os.mkdir(np_path)
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
mesh = UnitSquareMesh(Nx, Nx)
# ----------------------------------------------------------------------------------------------------------------------

# Define location of Neumann boundary: 0 for left, 1 for right
NeumannLocation = 0


class Neumann(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (abs(x[0]) < DOLFIN_EPS or abs(1 - x[0]) < DOLFIN_EPS)
        # return on_boundary and (near(x[0], 0.0))# or near(x[1],1.0))
        # if NeumannLocation == 0:
        #     return on_boundary and (abs(x[0]) < DOLFIN_EPS or abs(x[0] - 1) < DOLFIN_EPS)
        # elif NeumannLocation == 1:
        #     return on_boundary and abs(x[0] - 1) < DOLFIN_EPS


NeumannBD = Neumann()

boundary_function = MeshFunction("size_t", mesh, mesh.geometric_dimension() - 1)
boundary_function.set_all(0)
NeumannBD.mark(boundary_function, 1)
# ----------------------------------------------------------------------------------------------------------------------
dx = Measure('dx', domain=mesh)
ds = Measure('ds', domain=mesh, subdomain_data=boundary_function)
# ----------------------------------------------------------------------------------------------------------------------

# ud = Expression(('1', '1'), degree=1)
# f = Expression(('x[0]+1','x[0]+1'),degree=2)
f = Expression(('x[1]*(1-x[1])', '0'), degree=2)
# df = Expression(('x[1]','x[1]'),degree=2)
df = Expression(('x[1]*(1-x[1])', '0'), degree=2)
# ----------------------------------------------------------------------------------------------------------------------

P2 = VectorElement('CG', triangle, 2)
P1 = FiniteElement('CG', triangle, 1)
TH = MixedElement([P2, P1])
W = FunctionSpace(mesh, TH)


# ----------------------------------------------------------------------------------------------------------------------

def boundary_top(x, on_boundary):
    return on_boundary and 1 - x[1] < DOLFIN_EPS  # near(x[1],1.0)# DOLFIN_EPS or x[0] < 1 - DOLFIN_EPS)
    # return on_boundary and near(x[1],1.0)# DOLFIN_EPS or x[0] < 1 - DOLFIN_EPS)


# if NeumannLocation == 0:
#     pass
# elif NeumannLocation == 1:
#     return on_boundary and x[0] < 1 - DOLFIN_EPS

def boundary_bottom(x, on_boundary):
    return on_boundary and x[1] < DOLFIN_EPS
    # return on_boundary and near(x[1], 0.0)
    # if NeumannLocation == 0:
    #     return on_boundary and (x[0] > DOLFIN_EPS or x[0] < 1 - DOLFIN_EPS)
    # elif NeumannLocation == 1:
    #     return on_boundary and x[0] < 1 - DOLFIN_EPS


def boundary_right(x, on_boundary):
    return on_boundary and 1 - x[0] < DOLFIN_EPS
    # if NeumannLocation == 0:
    #     return on_boundary and (x[0] > DOLFIN_EPS or x[0] < 1 - DOLFIN_EPS)
    # elif NeumannLocation == 1:
    #     return on_boundary and x[0] < 1 - DOLFIN_EPS


def boundary(x, on_boundary):
    return on_boundary and (x[0] > DOLFIN_EPS and abs(1 - x[0]) > DOLFIN_EPS)


# bcs = [DirichletBC(W.sub(0), (0, 0), boundary_top), DirichletBC(W.sub(0), (0, 0), boundary_bottom), DirichletBC(W.sub(0), (0, 0), boundary_right)]
bcs = [DirichletBC(W.sub(0), (0, 0), boundary)]

# ----------------------------------------------------------------------------------------------------------------------
time_interval = np.linspace(t0, T, int(T / h))
ud1 = lambda t: 0.5 * (np.cos(np.pi * (t - 0.5)) - 1 - np.cos(np.pi))
u_d = np.zeros((K, int(T / h), mesh.geometric_dimension()))
for k in range(K):
    for i in range(len(time_interval)):
        u_d[k, i, 0] = ud1(time_interval[i])


# ----------------------------------------------------------------------------------------------------------------------

def solve_primal_ode(wSol):
    x = np.zeros((K, int(T / h), mesh.geometric_dimension()))

    x[:, 0, 0] = np.array([0.2 for i in range(K)])
    x[:, 0, 1] = np.linspace(0.2, 0.9, K)

    u_values_array = np.zeros((K, int(T / h), mesh.geometric_dimension()))
    for b_iter in range(K):
        for k in range(int(T/h)-1):
            # print("time: " + str(t_k))

            point = np.array([x[b_iter, k, :][0].item(), x[b_iter, k, :][1].item()])
            u_values = wSol.sub(0)(point)
            x[b_iter, k + 1, :] = x[b_iter, k, :] + h * u_values
            u_values_array[b_iter, k, :] = u_values
        u_values_array[b_iter, k+1, :] = wSol.sub(0)(np.array([x[b_iter, k+1, :][0].item(), x[b_iter, k+1, :][1].item()]))
    return x, u_values_array


def solve_adjoint_ode(wSol, grad_u, x):
    mu = np.zeros((K, int(T / h), mesh.geometric_dimension()))
    for b_iter in range(K):
        N = len(time_interval[1:])
        for k in range(N - 1, -1, -1):
            point = np.array([x[b_iter, k, :][0].item(), x[b_iter, k, :][1].item()])
            grad_u_values = grad_u(point)
            grad_u_matr = np.array([[grad_u_values[0].item(), grad_u_values[1].item()],
                                    [grad_u_values[2].item(), grad_u_values[3].item()]])

            u_values = wSol.sub(0)(point)
            A = (np.identity(2) + h * grad_u_matr.T)
            b_vec = mu[b_iter, k + 1, :] - h * grad_u_matr.T @ (u_values - u_d[b_iter, k, :])
            mu[b_iter, k, :] = np.linalg.solve(A, b_vec)
    return mu


# ----------------------------------------------------------------------------------------------------------------------
V_vec = TensorFunctionSpace(mesh, "Lagrange", 1)


# ----------------------------------------------------------------------------------------------------------------------
def J(u__, f_):
    # partA = assemble(0.5 * inner(u - u_d, u - u_d) * dx)
    partA = 0.5 * np.sum(np.sum(h * (np.linalg.norm(u__ - u_d, axis=2) ** 2), axis=1))
    partB = assemble(alpha * 0.5 * inner(f_, f_) * ds(int(1)))
    return partA + partB


# ----------------------------------------------------------------------------------------------------------------------
J_array = []
x_array = []
divs_u = []


# ----------------------------------------------------------------------------------------------------------------------
def grad_test(a, v, w, J0, gradj, iter):
    with open(np_path + f"grad_J_error_{iter}.txt", "w") as text_file:
        text_file.write("reduced Gradient j \t \t approximated gradient J \t Error \t \t \t h_i \n")
        # print("Gradient, one sided Approximation, Error, h")
        for k in range(3, 12):
            h_ = 10 ** (-k)
            F = a - inner(f + h_ * df, v) * ds(int(1))
            solve(F == 0, w, bcs)
            _, u_values_array = solve_primal_ode(w)
            gradapprox = (J(u_values_array, f + h_ * df) - J0) / h_
            # print(gradj, gradapprox, abs(gradj - gradapprox), h)
            # assemble(....)
            text_file.write(f" {gradj} \t {gradapprox} \t {abs(gradapprox - gradj)} \t {h_} \n")
        text_file.close()
    with open(np_path + f"grad_J_error_centered_{iter}.txt", "w") as text_file:
        text_file.write("reduced Gradient j \t \t approximated gradient J \t Error \t \t \t h_i \n")
        for k in range(3, 12):
            h_ = 10 ** (-k)
            F = a - inner(f + h_ * df, v) * ds(int(1))
            solve(F == 0, w, bcs)
            _, u_values_array = solve_primal_ode(w)
            jEvalRight = J(u_values_array, f + h_ * df)

            F = a - inner(f - h_ * df, v) * ds(int(1))
            solve(F == 0, w, bcs)
            _, u_values_array = solve_primal_ode(w)
            jEvalLeft = J(u_values_array, f - h_ * df)
            gradapprox = (jEvalRight - jEvalLeft) / (2 * h_)

            text_file.write(f" {gradj} \t {gradapprox} \t {abs(gradj - gradapprox)} \t {h_} \n")
        text_file.close()

# ----------------------------------------------------------------------------------------------------------------------
tau = 0.5
c = 1e-4

for i in range(num_steps):
    w = Function(W)
    w_test = TestFunction(W)

    u, p = split(w)
    v, q = split(w_test)

    a = (inner(grad(u), grad(v)) + inner(dot(grad(u), u), v) + div(u) * q + div(v) * p) * dx
    F = a - inner(f, v) * ds(int(1))

    solve(F == 0, w, bcs)
    # ----------------------------------------------------------------------------------------------------------------------

    grad_u = grad(w.sub(0))
    grad_u_proj = project(grad_u, V_vec)
    x, u_values_array = solve_primal_ode(w)
    mu = solve_adjoint_ode(w, grad_u_proj, x)

    x_array.append(x)
    # ----------------------------------------------------------------------------------------------------------------------

    zrSol = Function(W)
    zSol, rSol = split(zrSol)

    w_ad = TrialFunction(W)
    u_ad, p_ad = split(w_ad)
    vq_ad = TestFunction(W)
    v_ad, q_ad = split(vq_ad)

    aAdj = (inner(grad(u_ad), grad(v_ad)) + inner(grad(w.sub(0)) * v_ad, u_ad) + inner(grad(v) * w.sub(0), u_ad) + div(
        u_ad) * q_ad + div(v_ad) * p_ad) * dx
    FAdj = inner(Constant((0.0, 0.0)), v_ad) * dx

    A = assemble(aAdj)
    b = assemble(FAdj)

    # A, b = assemble_system(aAdj, FAdj, bcs)
    # u_values = np.zeros((K, int(T / h), mesh.geometric_dimension()))
    for buoy, points in enumerate(x):
        for k, point in enumerate(points):
            # u_values[buoy, k, :] = w.sub(0)(point)
            gamma = h * (u_d[buoy, k, :] - w.sub(0)(point) + mu[buoy, k, :])
            delta0 = PointSource(W.sub(0).sub(0), Point(point), gamma[0])
            delta1 = PointSource(W.sub(0).sub(1), Point(point), gamma[1])
            delta0.apply(b)
            delta1.apply(b)

    bcs[0].apply(A)
    bcs[0].apply(b)

    solve(A, zrSol.vector(), b)
    # ----------------------------------------------------------------------------------------------------------------------
    gradj = assemble(inner(alpha * f - zSol, df) * ds(int(1)))
    cond = - c * gradj
    if i == 0:
        J0 = J(u_values_array, f)
        #print(J0)
        #grad_test(a, v, w, J0, gradj, i)
    # ----------------------------------------------------------------------------------------------------------------------
    LR = 2
    while True:
        print("line search")
        J_old = J(u_values_array, f)
        w_ls = Function(W)
        w_test_ls = TestFunction(W)

        u_ls, p_ls = split(w_ls)
        v_ls, q_ls = split(w_test_ls)
        f_ls = f + LR * df
        a_ls = (inner(grad(u_ls), grad(v_ls)) + inner(dot(grad(u_ls), u_ls), v_ls) + div(u_ls) * q_ls + div(v_ls) * p_ls) * dx
        F_ls = a_ls - inner(f + LR * df, v_ls) * ds(int(1))

        solve(F_ls == 0, w_ls, bcs)
        # ----------------------------------------------------------------------------------------------------------------------

        grad_u_ls = grad(w_ls.sub(0))
        grad_u_proj_ls = project(grad_u_ls, V_vec)
        x_ls, u_values_array_ls = solve_primal_ode(w_ls)
        J_new = J(u_values_array_ls, f_ls)
        print(J_old)
        print(J_new)
        if J_old - J_new >= LR * cond:
            break
        LR = tau * LR
    print(LR)
    f = f - LR * (alpha * f - zSol)

    J_array.append(J(u_values_array, f))

    divs_u.append(sqrt(assemble(div(u) * div(u) * dx)))

    if i>0 and abs(J_array[i] - J_array[i-1]) < 1e-4:
        print("cost small enough")
        break

# ----------------------------------------------------------------------------------------------------------------------
with open(np_path + "u_divergence.txt", "w") as text_file:
    for i, div_u_val in enumerate(divs_u):
        text_file.write("div(u) \t \t \t i  \n")
        text_file.write(f" {div_u_val} \t {i} \n")
# ----------------------------------------------------------------------------------------------------------------------
# parameter output
with open(np_path + "variables.txt", "w") as text_file:
    text_file.write("t0: %s \n" % t0)
    text_file.write("T: %s \n" % T)
    text_file.write("dt: %s \n" % h)
    text_file.write("viscosity: %s \n" % viscosity)
    text_file.write("buoy count: %s \n" % K)
    text_file.write("LR: %s \n" % LR)
    text_file.write("gradient descent steps: %s \n" % num_steps)
# ----------------------------------------------------------------------------------------------------------------------


print("plotting")
plt.plot(J_array)
plt.savefig(f"{np_path}J.png")

plt.clf()

os.mkdir(np_path + "buoy_movements")
os.mkdir(np_path + "buoy_movements/frames")
for k, x_ in enumerate(x_array):
    color = plt.cm.rainbow(np.linspace(0, 1, K))
    for i, x_buoy in enumerate(x_):
        x_coord = x_buoy[:, 0]
        y_coord = x_buoy[:, 1]
        plt.xlim(0.0, 1)
        plt.ylim(0.0, 1)
        plt.plot([x_buoy[0, 0], x_buoy[0, 0] + 1 / (np.pi)], [x_buoy[0, 1], x_buoy[0, 1]], label="x0 and xT for u_D",
                 color=color[i], linewidth=8)
        plt.plot(x_coord, y_coord, label=f"buoy_{i}_movement", color="b")

    plt.savefig(f"{np_path}buoy_movements/frames/buoy_movement_{k}.png")
    plt.clf()

c = plot(u, title="u_field")
plt.colorbar(c)
plt.savefig(f"{np_path}u_field.png")
