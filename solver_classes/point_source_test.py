# Create a point source for Poisson problem
# Author: Jørgen S. Dokken
# SPDX-License-Identifier: MIT

from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
import ufl
import numpy as np
import scifem
import matplotlib.pyplot as plt

N = 40
domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
domain.name = "mesh"
tdim = domain.topology.dim
domain.topology.create_connectivity(tdim-1, tdim)

value_shape = (2,)
V = dolfinx.fem.functionspace(domain, ("Lagrange", 2, value_shape))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
dofs = dolfinx.fem.locate_dofs_topological(V, 1, boundary_facets)
u_bc = dolfinx.fem.Function(V)
u_bc.x.array[:] = 0
bc = dolfinx.fem.dirichletbc(u_bc, dofs)

b = dolfinx.fem.Function(V)
b.x.array[:] = 0

geom_dtype = domain.geometry.x.dtype
if domain.comm.rank == 0:
    points = np.array([[0.68, 0.362, 0],
                       [0.14, 0.213, 0],
                       [0.68, 0.362, 0],
                       [0.14, 0.213, 0]], dtype=geom_dtype)
else:
    points = np.zeros((0, 3), dtype=geom_dtype)

gamma = 1.
point_source = scifem.PointSource(V.sub(1), points, magnitude=gamma)
point_source.apply_to_vector(b)

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

uh = dolfinx.fem.Function(V)
ksp.solve(b.x.petsc_vec, uh.x.petsc_vec)
uh.x.scatter_forward()

#plt.plot(uh.x.array)
#plt.savefig("uh.png")

# plot
import pyvista
from IPython import embed; embed()
pyvista.start_xvfb()
topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
values = np.zeros((geometry.shape[0], 3), dtype=np.float64)
values[:, :len(uh)] = uh.x.array.real.reshape((geometry.shape[0], len(uh)))

function_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
function_grid["u"] = values
glyphs = function_grid.glyph(orient="u", factor=0.2)

domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim)
grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(domain, domain.topology.dim))

plotter = pyvista.Plotter()
plotter.add_mesh(grid, style="wireframe", color="k")
plotter.add_mesh(glyphs)
plotter.view_xy()
plotter.save_graphic("image.svg")
