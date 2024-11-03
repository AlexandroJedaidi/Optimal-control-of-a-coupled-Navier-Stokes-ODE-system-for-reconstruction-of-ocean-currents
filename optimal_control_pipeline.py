import typing
import os

import dolfinx.fem
import dolfinx.fem.petsc
import dolfinx.io
import dolfinx.mesh
import mpi4py.MPI
import numpy as np
import numpy.typing
import petsc4py.PETSc
import sympy
import ufl
# import viskex

import multiphenicsx.fem
import multiphenicsx.fem.petsc

from NonlinearBlockProblem import NonlinearBlockProblem
from plotting import plot_array

# ---------------------------------------------------------------------------------------------------------------------
experiment_number = 1
experiment_path = f'results/experiments/{experiment_number}'
os.mkdir(experiment_path)
np_path = f'results/experiments/{experiment_number}/arrays'
plot_path = f"results/experiments/{experiment_number}/plots"
os.mkdir(np_path)
os.mkdir(plot_path)
# ---------------------------------------------------------------------------------------------------------------------
mesh = dolfinx.mesh.create_unit_square(mpi4py.MPI.COMM_WORLD, 32, 32)

# Create connectivities required by the rest of the code
mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)


def bottom(x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.bool_]:
    """Condition that defines the bottom boundary."""
    return abs(x[1] - 0.) < np.finfo(float).eps  # type: ignore[no-any-return]


def left(x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.bool_]:
    """Condition that defines the left boundary."""
    return abs(x[0] - 0.) < np.finfo(float).eps  # type: ignore[no-any-return]


def top(x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.bool_]:
    """Condition that defines the top boundary."""
    return abs(x[1] - 1.) < np.finfo(float).eps  # type: ignore[no-any-return]


def right(x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.bool_]:
    """Condition that defines the right boundary."""
    return abs(x[0] - 1.) < np.finfo(float).eps  # type: ignore[no-any-return]


boundaries_entities = dict()
boundaries_values = dict()
for (boundary, boundary_id) in zip((bottom, left, top, right), (1, 2, 3, 4)):
    boundaries_entities[boundary_id] = dolfinx.mesh.locate_entities_boundary(
        mesh, mesh.topology.dim - 1, boundary)
    boundaries_values[boundary_id] = np.full(
        boundaries_entities[boundary_id].shape, boundary_id, dtype=np.int32)
boundaries_entities_unsorted = np.hstack(list(boundaries_entities.values()))
boundaries_values_unsorted = np.hstack(list(boundaries_values.values()))
boundaries_entities_argsort = np.argsort(boundaries_entities_unsorted)
boundaries_entities_sorted = boundaries_entities_unsorted[boundaries_entities_argsort]
boundaries_values_sorted = boundaries_values_unsorted[boundaries_entities_argsort]
boundaries = dolfinx.mesh.meshtags(
    mesh, mesh.topology.dim - 1,
    boundaries_entities_sorted, boundaries_values_sorted)
boundaries.name = "boundaries"

boundaries_134 = boundaries.indices[np.isin(boundaries.values, (1, 3, 4))]
boundaries_2 = boundaries.indices[boundaries.values == 2]

# Define associated measures
ds = ufl.Measure("ds", subdomain_data=boundaries)

# viskex.dolfinx.plot_mesh_tags(mesh, boundaries, "boundaries")

Y_velocity = dolfinx.fem.functionspace(mesh, ("Lagrange", 2, (mesh.geometry.dim,)))
Y_pressure = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
U = dolfinx.fem.functionspace(mesh, ("Lagrange", 2, (mesh.geometry.dim,)))
Q_velocity = Y_velocity.clone()
Q_pressure = Y_pressure.clone()

dofs_Y_velocity = np.arange(0, Y_velocity.dofmap.index_map.size_local + Y_velocity.dofmap.index_map.num_ghosts)
dofs_Y_pressure = np.arange(0, Y_pressure.dofmap.index_map.size_local + Y_pressure.dofmap.index_map.num_ghosts)
dofs_U = dolfinx.fem.locate_dofs_topological(U, boundaries.dim, boundaries_2)
dofs_Q_velocity = dofs_Y_velocity
dofs_Q_pressure = dofs_Y_pressure
restriction_Y_velocity = multiphenicsx.fem.DofMapRestriction(Y_velocity.dofmap, dofs_Y_velocity)
restriction_Y_pressure = multiphenicsx.fem.DofMapRestriction(Y_pressure.dofmap, dofs_Y_pressure)
restriction_U = multiphenicsx.fem.DofMapRestriction(U.dofmap, dofs_U)
restriction_Q_velocity = multiphenicsx.fem.DofMapRestriction(Q_velocity.dofmap, dofs_Q_velocity)
restriction_Q_pressure = multiphenicsx.fem.DofMapRestriction(Q_pressure.dofmap, dofs_Q_pressure)
restriction = [
    restriction_Y_velocity, restriction_Y_pressure, restriction_U, restriction_Q_velocity, restriction_Q_pressure]

(dv, dp) = (ufl.TrialFunction(Y_velocity), ufl.TrialFunction(Y_pressure))
(w, q) = (ufl.TestFunction(Y_velocity), ufl.TestFunction(Y_pressure))
du = ufl.TrialFunction(U)
r = ufl.TestFunction(U)
(dz, db) = (ufl.TrialFunction(Q_velocity), ufl.TrialFunction(Q_pressure))
(s, d) = (ufl.TestFunction(Q_velocity), ufl.TestFunction(Q_pressure))

(v, p) = (dolfinx.fem.Function(Y_velocity), dolfinx.fem.Function(Y_pressure))
u = dolfinx.fem.Function(U)
(z, b) = (dolfinx.fem.Function(Q_velocity), dolfinx.fem.Function(Q_pressure))

alpha = 1.e-5
x, y = sympy.symbols("x[0], x[1]")
psi_d = 10 * (1 - sympy.cos(0.8 * np.pi * x)) * (1 - sympy.cos(0.8 * np.pi * y)) * (1 - x) ** 2 * (1 - y) ** 2
v_d_x = sympy.lambdify([x, y], psi_d.diff(y, 1))
v_d_y = sympy.lambdify([x, y], - psi_d.diff(x, 1))
v_d = dolfinx.fem.Function(Y_velocity)
v_d.interpolate(lambda x: np.stack((v_d_x(x[0], x[1]), v_d_y(x[0], x[1])), axis=0))
nu = 0.1
ff = dolfinx.fem.Constant(mesh, tuple(petsc4py.PETSc.ScalarType(0) for _ in range(2)))
bc0 = np.zeros((2,), dtype=petsc4py.PETSc.ScalarType)

F = [nu * ufl.inner(ufl.grad(z), ufl.grad(w)) * ufl.dx
     + ufl.inner(z, ufl.grad(w) * v) * ufl.dx + ufl.inner(z, ufl.grad(v) * w) * ufl.dx
     - ufl.inner(b, ufl.div(w)) * ufl.dx + ufl.inner(v - v_d, w) * ufl.dx,
     - ufl.inner(ufl.div(z), q) * ufl.dx,
     alpha * ufl.inner(u, r) * ds(2) - ufl.inner(z, r) * ds(2),
     nu * ufl.inner(ufl.grad(v), ufl.grad(s)) * ufl.dx + ufl.inner(ufl.grad(v) * v, s) * ufl.dx
     - ufl.inner(p, ufl.div(s)) * ufl.dx - ufl.inner(ff, s) * ufl.dx - ufl.inner(u, s) * ds(2),
     - ufl.inner(ufl.div(v), d) * ufl.dx]
dF = [[ufl.derivative(F_i, u_j, du_j) for (u_j, du_j) in zip((v, p, u, z, b), (dv, dp, du, dz, db))] for F_i in F]
dF[3][3] = dolfinx.fem.Constant(mesh, petsc4py.PETSc.ScalarType(0)) * ufl.inner(dz, s) * (ds(1) + ds(3) + ds(4))
bdofs_Y_velocity_134 = dolfinx.fem.locate_dofs_topological(
    Y_velocity, mesh.topology.dim - 1, boundaries_134)
bdofs_Q_velocity_134 = dolfinx.fem.locate_dofs_topological(
    Q_velocity, mesh.topology.dim - 1, boundaries_134)
bc = [dolfinx.fem.dirichletbc(bc0, bdofs_Y_velocity_134, Y_velocity),
      dolfinx.fem.dirichletbc(bc0, bdofs_Q_velocity_134, Q_velocity)]

J = 0.5 * ufl.inner(v - v_d, v - v_d) * ufl.dx + 0.5 * alpha * ufl.inner(u, u) * ds(2)
J_cpp = dolfinx.fem.form(J)

# Create problem by extracting state forms from the optimality conditions
F_state = [ufl.replace(F[i], {
    s: w, d: q,
    u: dolfinx.fem.Constant(mesh, tuple(petsc4py.PETSc.ScalarType(0) for _ in range(2)))}) for i in (3, 4)]
dF_state = [[ufl.derivative(Fs_i, u_j, du_j) for (u_j, du_j) in zip((v, p), (dv, dp))] for Fs_i in F_state]
dF_state[1][1] = dolfinx.fem.Constant(mesh, petsc4py.PETSc.ScalarType(0)) * ufl.inner(dp, q) * ufl.dx
bc_state = [bc[0]]
problem_state = NonlinearBlockProblem(F_state, dF_state, (v, p), bc_state)
F_vec_state = dolfinx.fem.petsc.create_vector_block(problem_state._F)
dF_mat_state = dolfinx.fem.petsc.create_matrix_block(problem_state._dF)

# Solve
snes = petsc4py.PETSc.SNES().create(mesh.comm)
snes.setTolerances(max_it=20)
snes.getKSP().setType("preonly")
snes.getKSP().getPC().setType("lu")
snes.getKSP().getPC().setFactorSolverType("mumps")
snes.setObjective(problem_state.obj)
snes.setFunction(problem_state.F, F_vec_state)
snes.setJacobian(problem_state.dF, J=dF_mat_state, P=None)
snes.setMonitor(lambda _, it, residual: print(it, residual))
vp = problem_state.create_snes_solution()
snes.solve(None, vp)
problem_state.update_solutions(vp)  # TODO can this be safely removed?
vp.destroy()
snes.destroy()

J_uncontrolled = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(J_cpp), op=mpi4py.MPI.SUM)
print("Uncontrolled J =", J_uncontrolled)
assert np.isclose(J_uncontrolled, 0.1784542)

# viskex.dolfinx.plot_vector_field(v, "uncontrolled state velocity", glyph_factor=1e-1)

# viskex.dolfinx.plot_scalar_field(p, "uncontrolled state pressure")

# PYTEST_XFAIL_AND_SKIP_NEXT: ufl.derivative(adjoint of the trilinear term) introduces spurious conj(trial)
assert not np.issubdtype(petsc4py.PETSc.ScalarType, np.complexfloating)

# Create problem associated to the optimality conditions
problem = NonlinearBlockProblem(F, dF, (v, p, u, z, b), bc, restriction)
F_vec = multiphenicsx.fem.petsc.create_vector_block(problem._F, restriction=restriction)
dF_mat = multiphenicsx.fem.petsc.create_matrix_block(problem._dF, restriction=(restriction, restriction))

# Solve
snes = petsc4py.PETSc.SNES().create(mesh.comm)
snes.setTolerances(max_it=20)
snes.getKSP().setType("preonly")
snes.getKSP().getPC().setType("lu")
snes.getKSP().getPC().setFactorSolverType("mumps")
snes.setObjective(problem.obj)
snes.setFunction(problem.F, F_vec)
snes.setJacobian(problem.dF, J=dF_mat, P=None)
snes.setMonitor(lambda _, it, residual: print(it, residual))
vpuzb = problem.create_snes_solution()
snes.solve(None, vpuzb)
problem.update_solutions(vpuzb)  # TODO can this be safely removed?
vpuzb.destroy()
snes.destroy()

J_controlled = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(J_cpp), op=mpi4py.MPI.SUM)
print("Optimal J =", J_controlled)
assert np.isclose(J_controlled, 0.1249381)
# from IPython import embed
# embed()
v_arr = v.x.array
p_arr = p.x.array
u_arr = u.x.array
z_arr = z.x.array
b_arr = b.x.array
num_velocity_dofs = Y_velocity.dofmap.index_map_bs * Y_velocity.dofmap.index_map.size_global
num_pressure_dofs = Y_pressure.dofmap.index_map_bs * Y_velocity.dofmap.index_map.size_global
label = r"FEniCSx  ({0:d} dofs)".format(num_velocity_dofs + num_pressure_dofs)
plot_array(None, v_arr, label, "state_velocity", plot_path)
# viskex.dolfinx.plot_vector_field(v, "state velocity", glyph_factor=1e-1)

# viskex.dolfinx.plot_scalar_field(p, "state pressure")

# viskex.dolfinx.plot_vector_field(u, "control", glyph_factor=1e-1)

# viskex.dolfinx.plot_vector_field(z, "adjoint velocity", glyph_factor=1)

# viskex.dolfinx.plot_scalar_field(b, "adjoint pressure")
