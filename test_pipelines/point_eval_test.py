from dolfinx import mesh, fem, io
from ufl import *
from mpi4py import MPI
import ufl
import basix.ufl
import dolfinx
import numpy as np

domain = mesh.create_rectangle(MPI.COMM_WORLD, [(0.0, 0.0), (1.0, 1.0)],
                          (3,3), mesh.CellType.quadrilateral)

V = dolfinx.fem.functionspace(domain,("Lagrange", 2, (domain.geometry.dim,)))
u = dolfinx.fem.Function(V)


def u_values(x):
    values = np.zeros((2, x.shape[1]))
    # values[0, :] = np.where(np.isclose(x[0, :], 0.0), 1.0, np.where(np.isclose(x[0, :], 2.0), 1.0, 0.0))  # x-component
    # values[1, :] = np.where(np.isclose(x[0, :], 0.0), 0.0, 0.0)  # y-component
    values[0, :] = x[0,:]**2 + 2*x[1,:]**2
    return values


u.interpolate(u_values)
grad_u = ufl.grad(u)
U_grad_fp = dolfinx.fem.functionspace(domain,
                                      ("Lagrange", 2, (domain.geometry.dim, domain.geometry.dim)))
u_grad_expr = dolfinx.fem.Expression(grad_u, U_grad_fp.element.interpolation_points())
u_grad_fct = dolfinx.fem.Function(U_grad_fp)
u_grad_fct.interpolate(u_grad_expr)
bb_tree = dolfinx.geometry.bb_tree(domain, domain.topology.dim)
point = np.array([0.5, 0.5, 0])
cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, point)
colliding_cells = dolfinx.geometry.compute_colliding_cells(domain, cell_candidates, point).array
if len(colliding_cells) < 0:
    print("no colliding cells")
    from IPython import embed
    embed()
grad_u_values = u_grad_fct.eval(point, colliding_cells[0])
grad_u_matr = np.array([[grad_u_values[0].item(), grad_u_values[1].item()],
                        [grad_u_values[2].item(), grad_u_values[3].item()]])
u_values = u.eval(point, colliding_cells[0])
