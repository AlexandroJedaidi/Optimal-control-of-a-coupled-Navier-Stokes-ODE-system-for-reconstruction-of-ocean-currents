# Create a point source for Poisson problem
# Author: Jørgen S. Dokken
# SPDX-License-Identifier: MIT

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.fem.petsc
import numpy as np
import ufl


def compute_cell_contributions(V, points):
    # Determine what process owns a point and what cells it lies within
    mesh = V.mesh
    point_ownership_data = dolfinx.cpp.geometry.determine_point_ownership(
        mesh._cpp_object, points, 1e-6)
    owning_points = np.asarray(point_ownership_data.dest_points).reshape(-1, 3)
    cells = point_ownership_data.dest_cells

    # Pull owning points back to reference cell
    mesh_nodes = mesh.geometry.x
    cmap = mesh.geometry.cmap
    ref_x = np.zeros((len(cells), mesh.geometry.dim),
                     dtype=mesh.geometry.x.dtype)
    for i, (point, cell) in enumerate(zip(owning_points, cells)):
        geom_dofs = mesh.geometry.dofmap[cell]
        ref_x[i] = cmap.pull_back(point.reshape(-1, 3), mesh_nodes[geom_dofs])

    # Create expression evaluating a trial function (i.e. just the basis function)
    u = ufl.TrialFunction(V)
    num_dofs = V.dofmap.dof_layout.num_dofs * V.dofmap.bs
    if len(cells) > 0:
        # NOTE: Expression lives on only this communicator rank
        expr = dolfinx.fem.Expression(u, ref_x, comm=MPI.COMM_SELF)
        values = expr.eval(mesh, np.asarray(cells, dtype=np.int32))

        # Strip out basis function values per cell
        basis_values = values[:num_dofs:num_dofs*len(cells)]
    else:
        basis_values = np.zeros(
            (0, num_dofs), dtype=dolfinx.default_scalar_type)
    return cells, basis_values


N = 80
domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
domain.name = "mesh"
domain.topology.create_connectivity(1, 2)

V = dolfinx.fem.functionspace(domain, ("Lagrange", 1, (2,)))

facets = dolfinx.mesh.exterior_facet_indices(domain.topology)

if domain.comm.rank == 0:
    points = np.array([[0.68, 0.36, 0]], dtype=domain.geometry.x.dtype)
else:
    points = np.zeros((0, 3), dtype=domain.geometry.x.dtype)


u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
a_compiled = dolfinx.fem.form(a)


boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
dofs = dolfinx.fem.locate_dofs_topological(V, 1, boundary_facets)
u_bc = dolfinx.fem.Function(V)
u_bc.x.array[:] = 0
# dofs = dolfinx.fem.locate_dofs_topological(V, 1, facets)
# u_bc = dolfinx.fem.Constant(domain, 0.)
bc = dolfinx.fem.dirichletbc(u_bc, dofs)

b = dolfinx.fem.Function(V)
b.x.array[:] = 0
cells, basis_values = compute_cell_contributions(V, points)
for cell, basis_value in zip(cells, basis_values):
    from IPython import embed; embed()
    dofs = V.dofmap.cell_dofs(cell)
    print(len(b.x.array))
    b.x.array[dofs] += basis_value
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

with dolfinx.io.VTXWriter(domain.comm, "uh.bp", [uh], engine="BP4") as bp:
    bp.write(0.0)

import pyvista
print(uh)
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