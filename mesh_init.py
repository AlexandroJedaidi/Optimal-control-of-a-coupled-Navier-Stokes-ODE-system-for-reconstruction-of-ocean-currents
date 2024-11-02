import gmsh
import numpy as np
from mpi4py import MPI
from dolfinx.io import (VTXWriter, distribute_entity_data, gmshio)

# ---------------------------------------------------------------------------------------------------------------------
# mesh initialization used from the dolfinx tutorial and adjusted for our purposes
gmsh.initialize()

L = 1.2
H = 0.41
c_x = c_y = 0.2
r = 0.05


def create_mesh(gdim, obst=True):
    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    if mesh_comm.rank == model_rank:
        rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag=1)
        if obst:
            obstacle = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)

    if mesh_comm.rank == model_rank:
        if obst:
            fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, obstacle)])
        gmsh.model.occ.synchronize()

    fluid_marker = 1
    if mesh_comm.rank == model_rank:
        volumes = gmsh.model.getEntities(dim=gdim)
        assert (len(volumes) == 1)
        gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
        gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")

    inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2, 3, 4, 5
    inflow, outflow, walls, obstacle = [], [], [], []
    if mesh_comm.rank == model_rank:
        boundaries = gmsh.model.getBoundary(volumes, oriented=False)
        for boundary in boundaries:
            center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
            if np.allclose(center_of_mass, [0, H / 2, 0]):
                inflow.append(boundary[1])
            elif np.allclose(center_of_mass, [L, H / 2, 0]):
                outflow.append(boundary[1])
            elif np.allclose(center_of_mass, [L / 2, H, 0]) or np.allclose(center_of_mass, [L / 2, 0, 0]):
                walls.append(boundary[1])
            else:
                obstacle.append(boundary[1])

        gmsh.model.addPhysicalGroup(1, walls, wall_marker)
        gmsh.model.setPhysicalName(1, wall_marker, "Walls")
        gmsh.model.addPhysicalGroup(1, inflow, inlet_marker)
        gmsh.model.setPhysicalName(1, inlet_marker, "Inlet")
        gmsh.model.addPhysicalGroup(1, outflow, outlet_marker)
        gmsh.model.setPhysicalName(1, outlet_marker, "Outlet")
        gmsh.model.addPhysicalGroup(1, obstacle, obstacle_marker)
        gmsh.model.setPhysicalName(1, obstacle_marker, "Obstacle")

    # Create distance field from obstacle.
    # Add threshold of mesh sizes based on the distance field
    # LcMax -                  /--------
    #                      /
    # LcMin -o---------/
    #        |         |       |
    #       Point    DistMin DistMax
    res_min = r / 3
    if mesh_comm.rank == model_rank:
        distance_field = gmsh.model.mesh.field.add("Distance")
        threshold_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.25 * H)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", r)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2 * H)
        min_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
        gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

    if mesh_comm.rank == model_rank:
        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        #gmsh.option.setNumber("Mesh.MeshSizeMax", 0.03)  # define max mesh size
        gmsh.model.mesh.generate(gdim)
        gmsh.model.mesh.setOrder(2)
        gmsh.model.mesh.optimize("Netgen")

    mesh, _, ft = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
    ft.name = "Facet markers"
    gmsh.write("mesh.msh")
    return mesh, ft, inlet_marker, wall_marker, outlet_marker, obstacle_marker

