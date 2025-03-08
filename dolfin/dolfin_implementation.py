from fenics import *
import json
import numpy as np
import os

Nx = 32
alpha = 1e-2
experiment = 25
np_path = f"results/dolfin/experiments/{experiment}/"
os.mkdir(np_path)
with open("../parameters.json", "r") as file:
    parameters = json.load(file)
    t0 = parameters["t0"]
    T = parameters["T"]
    h = parameters["dt"]
    viscosity = parameters["viscosity"]
    K = parameters["buoy count"]
    mesh_boundary_x = parameters["mesh_boundary_x"]
    mesh_boundary_y = parameters["mesh_boundary_y"]

# set_log_level(LogLevel.WARNING)
mesh = UnitSquareMesh(Nx, Nx)

# Define location of Neumann boundary: 0 for left, 1 for right
NeumannLocation = 0


class Neumann(SubDomain):
    def inside(self, x, on_boundary):
        if NeumannLocation == 0:
            return on_boundary and (abs(x[0]) < DOLFIN_EPS )#or abs(1-x[0])<DOLFIN_EPS)
        elif NeumannLocation == 1:
            return on_boundary and abs(x[0] - 1) < DOLFIN_EPS


NeumannBD = Neumann()

boundary_function = MeshFunction("size_t", mesh, mesh.geometric_dimension() - 1)
boundary_function.set_all(0)
NeumannBD.mark(boundary_function, 1)

dx = Measure('dx', domain=mesh)
ds = Measure('ds', domain=mesh, subdomain_data=boundary_function)

# ud = Expression(('1', '1'), degree=1)
# f = Expression(('x[0]+1','x[0]+1'),degree=2)
f = Expression(('x[1]*(1-x[1])', '0'), degree=2)
# df = Expression(('x[1]','x[1]'),degree=2)
df = Expression(('x[1]*(1-x[1])', '0'), degree=2)

P2 = VectorElement('CG', triangle, 2)
P1 = FiniteElement('CG', triangle, 1)
TH = MixedElement([P2, P1])
W = FunctionSpace(mesh, TH)

w = Function(W)
u, p = split(w)

w_test = TestFunction(W)
v, q = split(w_test)


a = (inner(grad(u), grad(v)) + inner(dot(grad(u),u),v) + div(u) * q + div(v) * p) * dx
F = a - inner(f,v)* ds(int(1))


def boundary(x, on_boundary):
    if NeumannLocation == 0:
        return on_boundary and (x[0] > DOLFIN_EPS) #and abs(1-x[0]) > DOLFIN_EPS)
    elif NeumannLocation == 1:
        return on_boundary and x[0] < 1 - DOLFIN_EPS


bcs = [DirichletBC(W.sub(0), (0, 0), boundary)]



solve(F==0, w, bcs)


def solve_ode(wSol, grad_u):
    time_interval = np.linspace(t0, T, int(T / h))
    x = np.zeros((K, int(T / h), mesh.geometric_dimension()))
    ud1 = lambda t: 0.5 * (np.cos(np.pi * (t - 0.5)) - 1 - np.cos(np.pi))
    u_d = np.zeros((K, int(T / h), mesh.geometric_dimension()))
    for k in range(K):
        for i in range(len(time_interval)):
            u_d[k, i, 0] = ud1(time_interval[i])

    x[:, 0, 0] = np.array([0.2 for i in range(K)])
    x[:, 0, 1] = np.linspace(0.2, 0.9, K)

    mu = np.zeros((K, int(T / h), mesh.geometric_dimension()))
    u_values_array = np.zeros((K, int(T / h), mesh.geometric_dimension()))
    for b in range(K):
        for k, t_k in enumerate(time_interval[:-1]):
            # print("time: " + str(t_k))
            point = np.array([x[b, k, :][0].item(), x[b, k, :][1].item()])
            u_values = wSol.sub(0)(point)
            x[b, k + 1, :] = x[b, k, :] + h * u_values
            u_values_array[b,k :] = u_values

    for b in range(K):
        N = len(time_interval[1:])
        for k in range(N - 1, -1, -1):
            point = np.array([x[b, k+1, :][0].item(), x[b, k+1, :][1].item()])
            grad_u_values = grad_u(point)
            grad_u_matr = np.array([[grad_u_values[0].item(), grad_u_values[1].item()],
                                    [grad_u_values[2].item(), grad_u_values[3].item()]])

            u_values = wSol.sub(0)(point)
            A = (np.identity(2) + h * grad_u_matr.T)
            b_vec = mu[b, k + 1, :] - h * grad_u_matr.T @ (u_values - u_d[b, k, :])
            mu[b, k, :] = np.linalg.solve(A, b_vec)
            # mu[b, k, :] = mu[b, k + 1, :] - h * grad_u_matr.T @ (u_d[b, k + 1, :] - u_values - mu[b, k + 1, :])

    return x, mu, u_d, u_values_array

V_vec = TensorFunctionSpace(mesh, "Lagrange", 1)
grad_u = grad(w.sub(0))
grad_u_proj = project(grad_u, V_vec)
x, mu, u_d, u_values_array = solve_ode(w, grad_u_proj)

zrSol = Function(W)
zSol, rSol = split(zrSol)

w_ad = TrialFunction(W)
u_ad, p_ad = split(w_ad)
vq_ad = TestFunction(W)
v_ad, q_ad = split(vq_ad)

aAdj = (inner(grad(u_ad), grad(v_ad)) + inner(grad(w.sub(0)) * v_ad, u_ad) + inner(grad(v) * w.sub(0), u_ad) + div(u_ad) * q_ad + div(v_ad) * p_ad) * dx
FAdj = inner(Constant((0.0, 0.0)), v_ad) * dx

A = assemble(aAdj)
b = assemble(FAdj)

# A, b = assemble_system(aAdj, FAdj, bcs)
u_values = np.zeros((K, int(T / h), mesh.geometric_dimension()))
for buoy, points in enumerate(x):
    for k, point in enumerate(points):
        u_values[buoy, k, :] = w.sub(0)(point)
        gamma = h*(u_d[buoy, k, :] - w.sub(0)(point) + mu[buoy, k, :])
        delta0 = PointSource(W.sub(0).sub(0), Point(point), gamma[0])
        delta1 = PointSource(W.sub(0).sub(1), Point(point), gamma[1])
        delta0.apply(b)
        delta1.apply(b)

bcs[0].apply(A)
bcs[0].apply(b)
solve(A, zrSol.vector(), b)

def J(u, f):
    # partA = assemble(0.5 * inner(u - u_d, u - u_d) * dx)
    partA = 0.5 * np.sum(np.sum(h * (np.linalg.norm(u - u_d, axis=2) ** 2), axis=1))
    partB = assemble(alpha * 0.5 * inner(f, f) * ds(int(1)))
    return partA + partB


J0 = J(u_values_array, f)
print(J0)
gradj = assemble(inner(alpha * f - zSol, df) * ds(int(1)))


with open(np_path + f"grad_J_error_{0}.txt", "w") as text_file:
    text_file.write("reduced Gradient j \t \t approximated gradient J \t Error \t \t \t h_i \n")
#print("Gradient, one sided Approximation, Error, h")
    for k in range(3, 12):
        h_ = 10 ** (-k)
        F = a - inner(f + h_ * df, v) * ds(int(1))
        solve(F==0, w, bcs)
        _, _, _, u_values_array = solve_ode(w, grad_u_proj)
        gradapprox = (J(u_values_array, f + h_ * df) - J0) / h_
        # print(gradj, gradapprox, abs(gradj - gradapprox), h)
        # assemble(....)
        text_file.write(f" {gradj} \t {gradapprox} \t {abs(gradapprox - gradj)} \t {h_} \n")

with open(np_path + f"grad_J_error_centered_{0}.txt", "w") as text_file:
    text_file.write("reduced Gradient j \t \t approximated gradient J \t Error \t \t \t h_i \n")
    for k in range(3, 12):
        h_ = 10 ** (-k)
        F = a - inner(f + h_ * df, v) * ds(int(1))
        solve(F == 0, w, bcs)
        _, _, _, u_values_array = solve_ode(w, grad_u_proj)
        jEvalRight = J(u_values_array, f + h_ * df)

        F = a - inner(f - h_ * df, v) * ds(int(1))
        solve(F == 0, w, bcs)
        _, _, _, u_values_array = solve_ode(w, grad_u_proj)
        jEvalLeft = J(u_values_array, f - h_ * df)
        gradapprox= (jEvalRight - jEvalLeft)/(2*h_)

        text_file.write(f" {gradj} \t {gradapprox} \t {abs(gradj - gradapprox)} \t {h_} \n")



