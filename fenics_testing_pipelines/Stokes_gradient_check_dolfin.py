from fenics import *

Nx = 32
alpha = 1e-2
set_log_level(LogLevel.WARNING)
mesh = UnitSquareMesh(Nx, Nx)
# Define location of Neumann boundary: 0 for left, 1 for right
NeumannLocation = 0


class Neumann(SubDomain):
    def inside(self, x, on_boundary):
        if NeumannLocation == 0:
            return on_boundary and (abs(x[0]) < DOLFIN_EPS or abs(1-x[0])<DOLFIN_EPS)


NeumannBD = Neumann()
boundary_function = MeshFunction("size_t", mesh, mesh.geometric_dimension() - 1)
boundary_function.set_all(0)
NeumannBD.mark(boundary_function, 1)
dx = Measure('dx', domain=mesh)
ds = Measure('ds', domain=mesh, subdomain_data=boundary_function)
ud = Expression(('1', '1'), degree=1)
# f = Expression(('x[0]+1','x[0]+1'),degree=2)
f = Expression(('x[1]*(1-x[1])', '0'), degree=2)
# df = Expression(('x[1]','x[1]'),degree=2)
df = Expression(('x[1]*(1-x[1])', '0'), degree=2)
P2 = VectorElement('CG', triangle, 2)
P1 = FiniteElement('CG', triangle, 1)
TH = MixedElement([P2, P1])
W = FunctionSpace(mesh, TH)
u, p = split(TrialFunction(W))

v, q = split(TestFunction(W))
upSol = Function(W)
uSol, pSol = split(upSol)
a = (inner(grad(u), grad(v)) + div(u) * q + div(v) * p) * dx
# F = inner(f,v)*dx
F = inner(f, v) * ds(int(1))


def boundary(x, on_boundary):
    if NeumannLocation == 0:
        return on_boundary and (x[0] > DOLFIN_EPS and abs(1-x[0]) > DOLFIN_EPS)


bcs = [DirichletBC(W.sub(0), (0, 0), boundary)]


def J(u, f):
    partA = assemble(0.5 * inner(u - ud, u - ud) * dx)
    partB = assemble(alpha * 0.5 * inner(f, f) * ds(int(1)))
    return partA + partB


solve(a == F, upSol, bcs)

J0 = J(uSol, f)
zrSol = Function(W)
zSol, rSol = split(zrSol)
aAdj = (inner(grad(u), grad(v)) + div(u) * q + div(v) * p) * dx
FAdj = inner(uSol - ud, v) * dx
solve(aAdj == FAdj, zrSol, bcs)
gradj = assemble(inner(zSol + alpha * f, df) * ds(int(1)))

print("Gradient, one sided Approximation, Error, h")
for k in range(3, 12):
    h = 10 ** (-k)
    F = inner(f + h * df, v) * ds(int(1))
    solve(a == F, upSol, bcs)
    gradapprox = (J(uSol, f + h * df) - J0) / h
    print(gradj, gradapprox, abs(gradj - gradapprox), h)
    # assemble(....)

print("\n")
print("Gradient, symmetric Approximation, Error, h")
for k in range(3, 12):
    h = 10 ** (-k)
    F = inner(f + h * df, v) * ds(int(1))
    solve(a == F, upSol, bcs)
    jEvalRight = J(uSol, f + h * df)
    # gradapproxRight = (J(uSol,f+h*df)-J0)/h
    F = inner(f - h * df, v) * ds(int(1))
    solve(a == F, upSol, bcs)
    jEvalLeft = J(uSol, f - h * df)
    gradapprox = (jEvalRight - jEvalLeft) / (2 * h)
    print(gradj, gradapprox, abs(gradj - gradapprox), h)
print("\n")
# Check for divergence:
print("||div u||_L2 = ", sqrt(assemble(div(uSol) * div(uSol) * dx)))
