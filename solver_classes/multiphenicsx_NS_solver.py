import ufl
from basix.ufl import element, mixed_element
import dolfinx
import petsc4py
from solver_classes.NavierStokesProblem import NavierStokesProblem


def run_monolithic(mesh, vis, bcu, q_control, ds, inlet_marker) -> dolfinx.fem.Function:
    """Run standard formulation using a mixed function space."""
    # Function spaces
    U_el = element("CG", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
    P_el = element("CG", mesh.basix_cell(), 1)
    W_element = mixed_element([U_el, P_el])
    W = dolfinx.fem.functionspace(mesh, W_element)

    # Test and trial functions: monolithic
    vq = ufl.TestFunction(W)
    (v, q) = ufl.split(vq)
    dup = ufl.TrialFunction(W)
    up = dolfinx.fem.Function(W)
    (u, p) = ufl.split(up)

    # Variational forms
    F = (vis * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
         - ufl.inner(p, ufl.div(v)) * ufl.dx
         + ufl.inner(ufl.div(u), q) * ufl.dx
         - ufl.inner(q_control, v) * ds(inlet_marker))
    J = ufl.derivative(F, up, dup)

    # Create problem
    problem = NavierStokesProblem(F, J, up, bcu)
    F_vec = dolfinx.fem.petsc.create_vector(problem._F)
    J_mat = dolfinx.fem.petsc.create_matrix(problem._J)

    # Solve
    snes = petsc4py.PETSc.SNES().create(mesh.comm)
    snes.setTolerances(max_it=20)
    snes.getKSP().setType("preonly")
    snes.getKSP().getPC().setType("lu")
    snes.getKSP().getPC().setFactorSolverType("mumps")
    snes.setObjective(problem.obj)
    snes.setFunction(problem.F, F_vec)
    snes.setJacobian(problem.J, J=J_mat, P=None)
    snes.setMonitor(lambda _, it, residual: print(it, residual))
    up_copy = problem.create_snes_solution()
    snes.solve(None, up_copy)
    problem.update_solution(up_copy)  # TODO can this be safely removed?
    up_copy.destroy()
    snes.destroy()
    return up
