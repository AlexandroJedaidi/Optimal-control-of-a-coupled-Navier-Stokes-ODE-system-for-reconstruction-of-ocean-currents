# Imports
import os
import gmsh
import dolfinx
from petsc4py import PETSc
import ufl
from mpi4py import MPI
import json
from dolfinx.fem import (Constant, Function, functionspace,
                         assemble_scalar, dirichletbc, form, locate_dofs_topological, set_bc)
from ufl import (FacetNormal, Identity, Measure, TestFunction, TrialFunction,
                 as_vector, div, dot, inner, lhs, grad, nabla_grad, rhs, sym, system, SpatialCoordinate, inv,
                 sqrt, transpose, tr, TestFunctions, TrialFunctions, cos, pi, sin, dx)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_vector, create_matrix, set_bc)
import numpy as np
import numpy.typing as npt
import mesh_init
import solver_classes.Navier_stokes_solver
import solver_classes.ODE_solver
from basix.ufl import element, mixed_element
import matplotlib.pyplot as plt
#import imageio.v2 as imageio



def u_ex(x):
    sinx = sin(pi * x[0])
    siny = sin(pi * x[1])
    cosx = cos(pi * x[0])
    cosy = cos(pi * x[1])
    c_factor = 2 * pi * sinx * siny
    return c_factor * as_vector((cosy * sinx, -cosx * siny))


def p_ex(x):
    return sin(2 * pi * x[0]) * sin(2 * pi * x[1])

def source(x):
    u, p = u_ex(x), p_ex(x)
    return -div(grad(u)) + grad(p)

def create_bilinear_form(V, Q):
    u, p = TrialFunction(V), TrialFunction(Q)
    v, q = TestFunction(V), TestFunction(Q)
    a_uu = inner(grad(u), grad(v)) * dx
    a_up = inner(p, div(v)) * dx
    a_pu = inner(div(u), q) * dx
    return form([[a_uu, a_up], [a_pu, None]])

def create_linear_form(V, Q,qh,ds_, inlet_marker):
    v, q = TestFunction(V), TestFunction(Q)
    domain = V.mesh
    x = SpatialCoordinate(domain)
    f = source(x)
    return form([inner(f, v) * dx + inner(qh,v)*ds_(inlet_marker), inner(Constant(domain, 0.0), q) * dx])

def create_velocity_bc(V):
    domain = V.mesh
    g = Constant(domain, [0.0, 0.0])
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim - 1, tdim)
    bdry_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
    dofs = locate_dofs_topological(V, tdim - 1, bdry_facets)
    return [dirichletbc(g, dofs, V)]

def create_nullspace(rhs_form):
    null_vec = dolfinx.fem.petsc.create_vector_nest(rhs_form)
    null_vecs = null_vec.getNestSubVecs()
    null_vecs[0].set(0.0)
    null_vecs[1].set(1.0)
    null_vec.normalize()
    nsp = PETSc.NullSpace().create(vectors=[null_vec])
    return nsp

def create_preconditioner(Q, a, bcs):
    p, q = TrialFunction(Q), TestFunction(Q)
    a_p11 = form(inner(p, q) * dx)
    a_p = form([[a[0][0], None], [None, a_p11]])
    P = dolfinx.fem.petsc.assemble_matrix_nest(a_p, bcs)
    P.assemble()
    return P

def assemble_system(lhs_form, rhs_form, bcs):
    A = dolfinx.fem.petsc.assemble_matrix_nest(lhs_form, bcs=bcs)
    A.assemble()

    b = dolfinx.fem.petsc.assemble_vector_nest(rhs_form)
    dolfinx.fem.petsc.apply_lifting_nest(b, lhs_form, bcs=bcs)
    for b_sub in b.getNestSubVecs():
        b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    spaces = dolfinx.fem.extract_function_spaces(rhs_form)
    bcs0 = dolfinx.fem.bcs_by_block(spaces, bcs)
    dolfinx.fem.petsc.set_bc_nest(b, bcs0)
    return A, b

def create_block_solver(A, b, P, comm):
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A, P)
    ksp.setType("minres")
    ksp.setTolerances(rtol=1e-9)
    ksp.getPC().setType("fieldsplit")
    ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)

    nested_IS = P.getNestISs()
    ksp.getPC().setFieldSplitIS(("u", nested_IS[0][0]), ("p", nested_IS[0][1]))

    # Set the preconditioners for each block
    ksp_u, ksp_p = ksp.getPC().getFieldSplitSubKSP()
    ksp_u.setType("preonly")
    ksp_u.getPC().setType("gamg")
    ksp_p.setType("preonly")
    ksp_p.getPC().setType("jacobi")

    # Monitor the convergence of the KSP
    ksp.setFromOptions()
    return ksp

def assemble_scalarr(J, comm: MPI.Comm):
    scalar_form = form(J)
    local_J = dolfinx.fem.assemble_scalar(scalar_form)
    return comm.allreduce(local_J, op=MPI.SUM)


def compute_errors(u, p):
    domain = u.function_space.mesh
    x = SpatialCoordinate(domain)
    error_u = u - u_ex(x)
    H1_u = inner(error_u, error_u) * dx
    H1_u += inner(grad(error_u), grad(error_u)) * dx
    velocity_error = np.sqrt(assemble_scalarr(H1_u, domain.comm))

    error_p = -p - p_ex(x)
    L2_p = form(error_p * error_p * dx)
    pressure_error = np.sqrt(assemble_scalarr(L2_p, domain.comm))
    return velocity_error, pressure_error

def solve_stokes(u_element, p_element, domain, ds_, inlet_marker, qh):
    V = functionspace(domain, u_element)
    Q = functionspace(domain, p_element)

    lhs_form = create_bilinear_form(V, Q)
    rhs_form = create_linear_form(V, Q,qh, ds_, inlet_marker)

    bcs = create_velocity_bc(V)
    nsp = create_nullspace(rhs_form)
    A, b = assemble_system(lhs_form, rhs_form, bcs)
    assert nsp.test(A)
    A.setNullSpace(nsp)

    P = create_preconditioner(Q, lhs_form, bcs)
    ksp = create_block_solver(A, b, P, domain.comm)

    u, p = Function(V), Function(Q)
    w = PETSc.Vec().createNest([u.x.petsc_vec, p.x.petsc_vec])
    ksp.solve(b, w)
    assert ksp.getConvergedReason() > 0
    u.x.scatter_forward()
    p.x.scatter_forward()
    return compute_errors(u, p), u