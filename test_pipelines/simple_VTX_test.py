from dolfinx import mesh, fem, io
from ufl import *
from mpi4py import MPI
import ufl
import basix.ufl

domain = mesh.create_rectangle(MPI.COMM_WORLD, [(0.0, 0.0), (1.0, 1.0)],
                          (3,3), mesh.CellType.quadrilateral)

# create a Taylor Hood function space
velocity_element = basix.ufl.element("Lagrange", domain.ufl_cell(), degree=2)
pressure_element = basix.ufl.element("Lagrange", domain.ufl_cell(), degree=1)
combinedSpace = fem.functionspace(domain, velocity_element*pressure_element)

f = fem.Function(combinedSpace)

f1 = f.sub(1).collapse()
f1.name = "pressure"
file1 = io.VTXWriter(domain.comm, "output1.bp", f1, "BP4")
file1.write(0.0)
file1.close()

f2 = f.sub(0).collapse()
f2.name = "velocity"
file2 = io.VTXWriter(domain.comm, "output2.bp", f2, engine="BP4")
file2.write(0.0)
file2.close()