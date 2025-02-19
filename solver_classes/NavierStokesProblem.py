import ufl
import dolfinx
import typing
import petsc4py.PETSc
import petsc4py
import numpy as np

# Class for interfacing with SNES
class NavierStokesProblem:
    """Define a nonlinear problem, interfacing with SNES."""

    def __init__(  # type: ignore[no-any-unimported]
            self, F: ufl.Form, J: ufl.Form, solution: dolfinx.fem.Function,
            bcs: list[dolfinx.fem.DirichletBC], P: typing.Optional[ufl.Form] = None
    ) -> None:
        self._F = dolfinx.fem.form(F)
        self._J = dolfinx.fem.form(J)
        self._obj_vec = dolfinx.fem.petsc.create_vector(self._F)
        self._solution = solution
        self._bcs = bcs
        self._P = P

    def create_snes_solution(self) -> petsc4py.PETSc.Vec:  # type: ignore[no-any-unimported]
        """
        Create a petsc4py.PETSc.Vec to be passed to petsc4py.PETSc.SNES.solve.

        The returned vector will be initialized with the initial guess provided in `self._solution`.
        """
        x = self._solution.x.petsc_vec.copy()
        with x.localForm() as _x, self._solution.x.petsc_vec.localForm() as _solution:
            _x[:] = _solution
        return x

    def update_solution(self, x: petsc4py.PETSc.Vec) -> None:  # type: ignore[no-any-unimported]
        """Update `self._solution` with data in `x`."""
        x.ghostUpdate(addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
        with x.localForm() as _x, self._solution.x.petsc_vec.localForm() as _solution:
            _solution[:] = _x

    def obj(  # type: ignore[no-any-unimported]
            self, snes: petsc4py.PETSc.SNES, x: petsc4py.PETSc.Vec
    ) -> np.float64:
        """Compute the norm of the residual."""
        self.F(snes, x, self._obj_vec)
        return self._obj_vec.norm()  # type: ignore[no-any-return]

    def F(  # type: ignore[no-any-unimported]
            self, snes: petsc4py.PETSc.SNES, x: petsc4py.PETSc.Vec, F_vec: petsc4py.PETSc.Vec
    ) -> None:
        """Assemble the residual."""
        self.update_solution(x)
        with F_vec.localForm() as F_vec_local:
            F_vec_local.set(0.0)
        dolfinx.fem.petsc.assemble_vector(F_vec, self._F)
        dolfinx.fem.petsc.apply_lifting(F_vec, [self._J], [self._bcs], x0=[x], alpha=-1.0)
        F_vec.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(F_vec, self._bcs, x, -1.0)

    def J(  # type: ignore[no-any-unimported]
            self, snes: petsc4py.PETSc.SNES, x: petsc4py.PETSc.Vec, J_mat: petsc4py.PETSc.Mat,
            P_mat: petsc4py.PETSc.Mat
    ) -> None:
        """Assemble the jacobian."""
        J_mat.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(  # type: ignore[misc]
            J_mat, self._J, self._bcs, diagonal=1.0)  # type: ignore[arg-type]
        J_mat.assemble()
        if self._P is not None:
            P_mat.zeroEntries()
            dolfinx.fem.petsc.assemble_matrix(  # type: ignore[misc]
                P_mat, self._P, self._bcs, diagonal=1.0)  # type: ignore[arg-type]
            P_mat.assemble()