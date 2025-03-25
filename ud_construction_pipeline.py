from fenics import *
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import mshr
import os
from vedo.dolfin import plot as vplt

# ----------------------------------------------------------------------------------------------------------------------
# setup


# ----------------------------------------------------------------------------------------------------------------------
Nx = 32
experiment = 8
num_steps = 1
np_path = f"results/dolfin/OCP/ud_construction/{experiment}/"
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
# ----------------------------------------------------------------------------------------------------------------------
# mesh = UnitSquareMesh(Nx, Nx)
left_x = 0.0
left_y = 2.0
right_x = 2.0
right_y = 2.0
mesh = RectangleMesh(Point(left_x, left_x), Point(right_x, right_y), Nx, Nx)
Nx_t = 50
# rect1 = mshr.Rectangle(Point(0.0, 0.0), Point(0.5, 2.0))
# rect2 = mshr.Rectangle(Point(0.5, 0.0), Point(1.5, 3.0))
# rect3 = mshr.Rectangle(Point(1.5, 0.0), Point(2.0, 1.8))
# mesh = mshr.generate_mesh(rect1 + rect2 + rect3, Nx_t)
plt.title(r"discretized domain $\Omega_h$")
plt.xlabel("x")
plt.ylabel("y")
plot(mesh)
plt.savefig(f"{np_path}mesh.png")
plt.clf()
mesh_boundary = [[[0.0, 2.0], [0.0, 0.0]],
                 [[0.0, 0.0], [0.0, 2.0]],
                 [[0.0, 2.0], [2.0, 2.0]],
                 [[2.0, 2.0], [2.0, 0.0]]]


# ----------------------------------------------------------------------------------------------------------------------

class Neumann(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (abs(x[0]) < DOLFIN_EPS or (abs(2.0 - x[0]) < DOLFIN_EPS))


NeumannBD = Neumann()

boundary_function = MeshFunction("size_t", mesh, mesh.geometric_dimension() - 1)
#boundary_function.set_all(0)
#NeumannBD.mark(boundary_function, 1)
# ----------------------------------------------------------------------------------------------------------------------
dx = Measure('dx', domain=mesh)
ds = Measure('ds', domain=mesh, subdomain_data=boundary_function)
# ----------------------------------------------------------------------------------------------------------------------
f = Expression(('0', '0'), degree=2)
df = Expression(('0.1', '0.1'), degree=2)
# F1 = Expression("pi*sin(pi*x[0])*sin(pi*x[1]) - nu*pi*pi*(-cos(pi*x[0])*sin(pi*x[1]))", nu=viscosity, degree=2)
# F2 = Expression("-pi*cos(pi*x[0])*cos(pi*x[1]) - nu*pi*pi*(sin(pi*x[0])*cos(pi*x[1]))", nu=viscosity, degree=2)

f_x = Expression("alpha * sin(pi*x[0]) * cos(pi*x[1])", alpha=1.0, degree=2)
f_y = Expression("-alpha * cos(pi*x[0]) * sin(pi*x[1])", alpha=1.0, degree=2)
# f_x = Expression("1.0", alpha=1.0, degree=2)
# f_y = Expression("0.0", alpha=1.0, degree=2)
# F_rhs = as_vector([F1, F2])
F_rhs = as_vector([f_x, f_y])

# ----------------------------------------------------------------------------------------------------------------------
P2 = VectorElement('CG', triangle, 2)
P1 = FiniteElement('CG', triangle, 1)
TH = MixedElement([P2, P1])
W = FunctionSpace(mesh, TH)
# ----------------------------------------------------------------------------------------------------------------------
inflow = Expression(("-cos(pi*x[0])*sin(pi*x[1])", "sin(pi*x[0])*cos(pi*x[1])"), degree=2)
# inflow = Expression(("1.0", "0.0"), degree=2)
zero_p = Constant(0)
zero_velocity = Constant((0.0, 0.0))
noslip = "near(x[1], 0) || near(x[1], 2) || near(x[1], 1.8) || near(x[1], 3)"
bc_u_top_bottom = DirichletBC(W.sub(0), zero_velocity, noslip)

# Apply analytical inflow condition on left (x = 0) and right (x = 1) boundaries
inflow_outflow = "near(x[0], 0) || near(x[0], 2)"
bc_u_inflow_outflow = DirichletBC(W.sub(0), inflow, inflow_outflow)
bc_p = DirichletBC(W.sub(1), zero_p, "x[0] < DOLFIN_EPS")
bcs = [bc_u_top_bottom, bc_u_inflow_outflow, bc_p]
# ----------------------------------------------------------------------------------------------------------------------
time_interval = np.linspace(t0, T, int(T / h))
u_d = np.zeros((K, int(T / h), mesh.geometric_dimension()))
x_d1 = np.zeros_like(time_interval)
x_d2 = np.zeros_like(time_interval)


# ----------------------------------------------------------------------------------------------------------------------

def solve_primal_ode(wSol):
    x = np.zeros((K, int(T / h), mesh.geometric_dimension()))
    x[:, 0, 0] = [0.25, 1.75, 0.5, 1.5]  # np.array([1.0 for i in range(K)])
    x[:, 0, 1] = [1.25, 0.5, 1.6, 0.3]  # np.linspace(0.5, 1.5, K)

    u_values_array = np.zeros((K, int(T / h), mesh.geometric_dimension()))
    for b_iter in range(K):
        for k in range(int(T / h) - 1):
            point = np.array([x[b_iter, k, :][0].item(), x[b_iter, k, :][1].item()])
            try:
                u_values = wSol.sub(0)(point)
            except:
                print(b_iter)
                print("cant evaluate at")
                print(point)
                c = plot(wSol.sub(0), title="u_field")
                plt.colorbar(c)
                plt.savefig(f"{np_path}u_field.png")
                plt.clf()
            x[b_iter, k + 1, :] = x[b_iter, k, :] + h * u_values
            u_values_array[b_iter, k, :] = u_values
        u_values_array[b_iter, k + 1, :] = wSol.sub(0)(
            np.array([x[b_iter, k + 1, :][0].item(), x[b_iter, k + 1, :][1].item()]))
    return x, u_values_array





# ----------------------------------------------------------------------------------------------------------------------
J_array = []
x_array = []
divs_u = []
# ----------------------------------------------------------------------------------------------------------------------
# solving primal NS
w = Function(W)
w_test = TestFunction(W)

u, p = split(w)
v, q = split(w_test)
n = FacetNormal(mesh)
a = (inner(grad(u), grad(v)) + inner(dot(grad(u), u), v) + div(u) * q + div(v) * p) * dx - 0.5 * (
    dot(dot(u, n) * u, v)) * ds(int(1))
F = a - inner(F_rhs, v) * ds(int(1))

solve(F == 0, w, bcs)
# ----------------------------------------------------------------------------------------------------------------------
# solving primal and adjoint ODE
x, u_values_array = solve_primal_ode(w)
x_array.append(x)

divs_u.append(sqrt(assemble(div(u) * div(u) * dx)))

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
    text_file.write("t0: %s \n" % t0)
    text_file.write("T: %s \n" % T)
    text_file.write("dt: %s \n" % h)
    text_file.write("viscosity: %s \n" % viscosity)
    text_file.write("buoy count: %s \n" % K)
    text_file.write("LR: %s \n" % LR)
    text_file.write("gradient descent steps: %s \n" % num_steps)
# ----------------------------------------------------------------------------------------------------------------------
# plotting
print("plotting")
plt.clf()

os.mkdir(np_path + "buoy_movements")
os.mkdir(np_path + "paraview")
os.mkdir(np_path + "buoy_movements/frames")

for k, x_ in enumerate(x_array):
    color = plt.cm.rainbow(np.linspace(0, 1, K))
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title(r"Buoy movement result $x$")
    for i, x_buoy in enumerate(x_):
        x_coord = x_buoy[:, 0]
        y_coord = x_buoy[:, 1]
        # plt.xlim(0.0, 3)
        # plt.ylim(0.0, 3)
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.plot(x_d1[i], x_d2[i], label=r"$x_d$", color="black", linewidth=2)
        plt.plot(x_coord, y_coord, label=r"$x$ for buoy" + f"{k + 1}", color="b")

    for line in mesh_boundary:
        plt.plot(line[0], line[1], color="gray")
    plt.legend(loc="upper right")
    plt.savefig(f"{np_path}buoy_movements/frames/buoy_movement_{k}.png")
    plt.clf()

c = plot(u, title=r"Velocity field $u$")
plt.colorbar(c)
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.savefig(f"{np_path}u_field.png")
plt.clf()


u,p = w.split()

xdmf_file = XDMFFile(f"{np_path}paraview/velocity.xdmf")
xdmf_file.write(u)

for k, x_ in enumerate(x_array):
    color = plt.cm.rainbow(np.linspace(0, 1, K))
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title(r"Buoy movement result $x$")
    for i, x_buoy in enumerate(x_):
        x_coord = x_buoy[:, 0]
        y_coord = x_buoy[:, 1]
        plt.xlim(0.0, 2)
        plt.ylim(0.0, 2)
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.plot(x_coord, y_coord, label=r"$x$ for buoy"+ f"{k+1}", color="b")

    # for line in mesh_boundary:
    #     plt.plot(line[0], line[1], color="gray")
    plt.legend(loc="upper right")
    plt.savefig(f"{np_path}buoy_movements/frames/buoy_movement_{k}.png")
    plt.clf()

with open(f"{np_path}u_d_array.npy", "wb") as np_writer:
    np.save(np_writer, u_values_array)

with open(f"{np_path}x_0_array.npy", "wb") as np_writer:
    np.save(np_writer, x)
