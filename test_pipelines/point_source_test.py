from dolfinx import mesh, fem, io
from mpi4py import MPI
import ufl
import basix.ufl
import dolfinx
import numpy as np
from ufl import (FacetNormal, Identity, Measure, TestFunction, TrialFunction,
                 as_vector, div, dot, inner, lhs, grad, nabla_grad, rhs, sym, system, SpatialCoordinate, inv,
                 sqrt, transpose, tr, ds, TestFunctions, TrialFunctions)
from basix.ufl import element, mixed_element
from dolfinx.fem import (Constant, Function, functionspace,
                         assemble_scalar, dirichletbc, form, locate_dofs_topological, set_bc)
import scifem
from petsc4py import PETSc

domain = mesh.create_rectangle(MPI.COMM_WORLD, [(0.0, 0.0), (1.0, 1.0)],
                               (3, 3), mesh.CellType.quadrilateral)
tdim = domain.topology.dim
domain.topology.create_connectivity(tdim - 1, tdim)

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
b = Function(U)
# b, _ = bp.split()
b.x.array[:] = 0

geom_dtype = domain.geometry.x.dtype
points = np.array([[0.25, 0.25, 0], [0.5, 0.5, 0], [0.75, 0.75, 0]], dtype=geom_dtype)
bb_tree = dolfinx.geometry.bb_tree(domain, domain.topology.dim)
cells = []
points_on_proc = []
cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, points)
colliding_cells = dolfinx.geometry.compute_colliding_cells(domain, cell_candidates, points)
for i, point in enumerate(points):
    if len(colliding_cells.links(i)) > 0:
        points_on_proc.append(point)
        cells.append(colliding_cells.links(i)[0])

# u_values = u.eval(points_on_proc, cells)
u_values = np.array([[1, 2.],
                     [1, 2.],
                     [1, 2.]])

ps1 = scifem.PointSource(U.sub(0), points, magnitude=u_values[:, 0])
ps2 = scifem.PointSource(U.sub(1), points, magnitude=u_values[:, 1])
ps1.apply_to_vector(b)
ps2.apply_to_vector(b)
b_val = b.eval(points_on_proc, cells)
from IPython import embed; embed()
boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
dofs = dolfinx.fem.locate_dofs_topological(U, 1, boundary_facets)
u_bc = dolfinx.fem.Function(U)
u_bc.x.array[:] = 0
bc = dolfinx.fem.dirichletbc(u_bc, dofs)
u_ = ufl.TrialFunction(U)
v = ufl.TestFunction(U)
a = ufl.inner(ufl.grad(u_), ufl.grad(v)) * ufl.dx
a_compiled = dolfinx.fem.form(a)
dolfinx.fem.petsc.apply_lifting(b.x.petsc_vec, [a_compiled], [[bc]])
b.x.scatter_reverse(dolfinx.la.InsertMode.add)
dolfinx.fem.petsc.set_bc(b.x.petsc_vec, [bc])
b.x.scatter_forward()
A = dolfinx.fem.petsc.assemble_matrix(a_compiled, bcs=[bc])
A.assemble()

ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.PREONLY)
ksp.getPC().setType(PETSc.PC.Type.LU)
ksp.getPC().setFactorSolverType("mumps")
uh = dolfinx.fem.Function(U)
ksp.solve(b.x.petsc_vec, uh.x.petsc_vec)
uh.x.scatter_forward()

vtx_u = dolfinx.io.VTXWriter(domain.comm, "_u.bp", uh, engine="BP4")
vtx_u.write(0.0)
vtx_u.close()

from IPython import embed;

embed()
# def test(x):
#     return np.isclose(x[0], points[0][0]) & np.isclose(x[1], points[0][1])
# dof_point_1 = dolfinx.fem.locate_dofs_geometrical(U, test)
# dof_point_2 = dolfinx.fem.locate_dofs_geometrical(U, lambda x: np.isclose(x[0], points[1][0]) & np.isclose(x[1], points[1][1]))
# from IPython import embed; embed()
