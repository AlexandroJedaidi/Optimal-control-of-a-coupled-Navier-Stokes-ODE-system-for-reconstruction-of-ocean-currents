# Imports
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
                 sqrt, transpose, tr)
import numpy as np
import numpy.typing as npt
import mesh_init
import solver_classes.Navier_stokes_solver
import solver_classes.ODE_solver

def transfer_meshtags_to_submesh(
    mesh: dolfinx.mesh.Mesh,
    entity_tag: dolfinx.mesh.MeshTags,
    submesh: dolfinx.mesh.Mesh,
    sub_vertex_to_parent: npt.NDArray[np.int32],
    sub_cell_to_parent: npt.NDArray[np.int32]
)-> tuple[dolfinx.mesh.MeshTags, npt.NDArray[np.int32]]:
    """
    Transfer a meshtag from a parent mesh to a sub-mesh.

    Args:
        mesh: Mesh containing the meshtags
        entity_tag: The meshtags object to transfer
        submesh: The submesh to transfer the `entity_tag` to
        sub_to_vertex_map: Map from each vertex in `submesh` to the corresponding
            vertex in the `mesh`
        sub_cell_to_parent: Map from each cell in the `submesh` to the corresponding
            entity in the `mesh`
    Returns:
        The entity tag defined on the submesh, and a map from the entities in the
        `submesh` to the entities in the `mesh`.

    """

    tdim = mesh.topology.dim
    cell_imap = mesh.topology.index_map(tdim)
    num_cells = cell_imap.size_local + cell_imap.num_ghosts
    mesh_to_submesh = np.full(num_cells, -1)
    mesh_to_submesh[sub_cell_to_parent] = np.arange(
        len(sub_cell_to_parent), dtype=np.int32
    )
    sub_vertex_to_parent = np.asarray(sub_vertex_to_parent)

    submesh.topology.create_connectivity(entity_tag.dim, 0)

    num_child_entities = (
        submesh.topology.index_map(entity_tag.dim).size_local
        + submesh.topology.index_map(entity_tag.dim).num_ghosts
    )
    submesh.topology.create_connectivity(submesh.topology.dim, entity_tag.dim)

    c_c_to_e = submesh.topology.connectivity(submesh.topology.dim, entity_tag.dim)
    c_e_to_v = submesh.topology.connectivity(entity_tag.dim, 0)

    child_markers = np.full(num_child_entities, 0, dtype=np.int32)

    mesh.topology.create_connectivity(entity_tag.dim, 0)
    mesh.topology.create_connectivity(entity_tag.dim, mesh.topology.dim)
    p_f_to_v = mesh.topology.connectivity(entity_tag.dim, 0)
    p_f_to_c = mesh.topology.connectivity(entity_tag.dim, mesh.topology.dim)
    sub_to_parent_entity_map = np.full(num_child_entities, -1, dtype=np.int32)
    for facet, value in zip(entity_tag.indices, entity_tag.values):
        facet_found = False
        for cell in p_f_to_c.links(facet):
            if facet_found:
                break
            if (child_cell := mesh_to_submesh[cell]) != -1:
                for child_facet in c_c_to_e.links(child_cell):
                    child_vertices = c_e_to_v.links(child_facet)
                    child_vertices_as_parent = sub_vertex_to_parent[child_vertices]
                    is_facet = np.isin(
                        child_vertices_as_parent, p_f_to_v.links(facet)
                    ).all()
                    if is_facet:
                        child_markers[child_facet] = value
                        facet_found = True
                        sub_to_parent_entity_map[child_facet] = facet
    tags = dolfinx.mesh.meshtags(
        submesh,
        entity_tag.dim,
        np.arange(num_child_entities, dtype=np.int32),
        child_markers,
    )
    tags.name = entity_tag.name
    return tags, sub_to_parent_entity_map

# ----------------------------------------------------------------------------------------------------------------------
experiment_number = 2
# Discretization parameters
with open("parameters.json", "r") as file:
    parameters = json.load(file)
    t0 = parameters["t0"]
    T = parameters["T"]
    dt = parameters["dt"]
    vis = parameters["viscosity"]
    K = parameters["buoy count"]
    mu = parameters["LR"]

    # num_steps = int(T / dt)
# ----------------------------------------------------------------------------------------------------------------------
# Mesh
gmsh.initialize()
gdim = 2
# mesh, ft, inlet_marker, wall_marker = mesh_init.create_mesh(gdim)
mesh, ft, inlet_marker, wall_marker, outlet_marker = mesh_init.create_pipe_mesh(gdim)
# ----------------------------------------------------------------------------------------------------------------------
NS_instance = solver_classes.Navier_stokes_solver.NavierStokes(mesh, ft, inlet_marker, wall_marker, outlet_marker,
                                                               experiment_number)
ODE_instance = solver_classes.ODE_solver.ODE(mesh, ft, inlet_marker, wall_marker)
Q = dolfinx.fem.functionspace(mesh, ("Lagrange", 2, (mesh.geometry.dim,)))
num_steps = 1000
q = Function(Q)
#q.x.array[:] = 0.1
for i in range(num_steps):
    w = NS_instance.state_solving_step(q)
    u, p = w.split()
    x, h, N, u_d = ODE_instance.ode_solving_step(u)
    lam_2 = ODE_instance.adjoint_ode_solving_step(u)
    w_adj = NS_instance.adjoint_state_solving_step(u, lam_2, x, h, N, u_d)
    u_adj, p_adj = w_adj.split()
    from IPython import embed
    embed()
    grad_j = q.x.array - u_adj.x.array

    q.x.array[:] = (1-mu)*q.x.array - mu * u_adj.x.array

