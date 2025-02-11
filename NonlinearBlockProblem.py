import typing

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


class NonlinearBlockProblem:
    """Define a nonlinear problem, interfacing with SNES."""

    def __init__(  # type: ignore[no-any-unimported]
            self, F: list[ufl.Form], dF: list[list[ufl.Form]],
            solutions: tuple[dolfinx.fem.Function, ...], bcs: list[dolfinx.fem.DirichletBC],
            restriction: typing.Optional[list[multiphenicsx.fem.DofMapRestriction]] = None
    ) -> None:
        self._F = dolfinx.fem.form(F)
        self._dF = dolfinx.fem.form(dF)
        self._obj_vec = multiphenicsx.fem.petsc.create_vector_block(self._F, restriction)
        self._solutions = solutions
        self._bcs = bcs
        self._restriction = restriction

    def create_snes_solution(self) -> petsc4py.PETSc.Vec:  # type: ignore[no-any-unimported]
        """
        Create a petsc4py.PETSc.Vec to be passed to petsc4py.PETSc.SNES.solve.

        The returned vector will be initialized with the initial guesses provided in `self._solutions`,
        properly stacked together and restricted in a single block vector.
        """
        x = multiphenicsx.fem.petsc.create_vector_block(self._F, restriction=self._restriction)
        with multiphenicsx.fem.petsc.BlockVecSubVectorWrapper(
                x, [c.function_space.dofmap for c in self._solutions], self._restriction) as x_wrapper:
            for x_wrapper_local, component in zip(x_wrapper, self._solutions):
                with component.x.petsc_vec.localForm() as component_local:
                    x_wrapper_local[:] = component_local
        return x

    def update_solutions(self, x: petsc4py.PETSc.Vec) -> None:  # type: ignore[no-any-unimported]
        """Update `self._solutions` with data in `x`."""
        x.ghostUpdate(addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
        with multiphenicsx.fem.petsc.BlockVecSubVectorWrapper(
                x, [c.function_space.dofmap for c in self._solutions], self._restriction) as x_wrapper:
            for x_wrapper_local, component in zip(x_wrapper, self._solutions):
                with component.x.petsc_vec.localForm() as component_local:
                    component_local[:] = x_wrapper_local

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
        self.update_solutions(x)
        with F_vec.localForm() as F_vec_local:
            F_vec_local.set(0.0)
        multiphenicsx.fem.petsc.assemble_vector_block(  # type: ignore[misc]
            F_vec, self._F, self._dF, self._bcs, x0=x, alpha=-1.0,
            restriction=self._restriction, restriction_x0=self._restriction)

    def dF(  # type: ignore[no-any-unimported]
            self, snes: petsc4py.PETSc.SNES, x: petsc4py.PETSc.Vec, dF_mat: petsc4py.PETSc.Mat,
            _: petsc4py.PETSc.Mat
    ) -> None:
        """Assemble the jacobian."""
        dF_mat.zeroEntries()
        if self._restriction is None:
            restriction = None
        else:
            restriction = (self._restriction, self._restriction)
        multiphenicsx.fem.petsc.assemble_matrix_block(
            dF_mat, self._dF, self._bcs, diagonal=1.0, restriction=restriction)  # type: ignore[arg-type]
        dF_mat.assemble()
