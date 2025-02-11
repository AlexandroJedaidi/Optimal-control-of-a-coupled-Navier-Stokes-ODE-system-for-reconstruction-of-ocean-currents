import sys
import dolfinx
import pyvista
import numpy as np
from basix.ufl import element
from dolfinx.fem import functionspace
import matplotlib.pyplot as plt
import time

# ---------------------------------------------------------------------------------------------------------------------
# pipeline to create videos for displaying (partial) results
NUMBER_X: int = 1
NUMBER_Y: int = 1

CANVAS_WIDTH: int = 10
CANVAS_HEIGHT: int = 10


def plotting_gif(list, fs, path, var_name):
    plotter = pyvista.Plotter()
    plotter.open_movie(path)
    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(fs)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    grid.point_data[var_name] = list[0]
    warped = grid.warp_by_scalar(var_name, factor=0.5)
    plotter.add_mesh(warped, show_edges=True, clim=[np.min(list), np.max(list)])
    for sol in list:
        new_warped = grid.warp_by_scalar(var_name, factor=0.1)
        warped.points[:, :] = new_warped.points
        warped.point_data[var_name][:] = sol
        plotter.write_frame()
    plotter.close()


def plotting_2d_gif(list, fs, path, var_name):
    plotter = pyvista.Plotter()
    plotter.open_movie(path)
    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(fs)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    grid.point_data[var_name] = list[0]
    grid.set_active_scalars(var_name)
    plotter.add_mesh(grid, show_edges=True, clim=[np.min(list), np.max(list)])
    plotter.view_xy()

    for sol in list:
        grid.point_data[var_name][:] = sol
        plotter.write_frame()
    plotter.close()


def streamlines_animation(data, fluid_mesh, mesh):
    line_streamlines = fluid_mesh.streamlines(
        pointa=(0, -5, 0),
        pointb=(0, 5, 0),
        n_points=25,
        max_time=100.0,
        compute_vorticity=False,  # vorticity already exists in dataset
    )

    clim = [0, 20]
    camera_position = [(7, 0, 20.0), (7, 0.0, 0.0), (0.0, 1.0, 0.0)]

    p = pyvista.Plotter()
    for i in range(1, len(mesh)):
        p.add_mesh(mesh[i], color='k')
    p.add_mesh(line_streamlines.tube(radius=0.05), scalars="vorticity_mag", clim=clim)
    p.view_xy()
    p.show(cpos=camera_position)


def plot_array(x, y, label, title, plot_path):
    fig = plt.figure(figsize=(25, 8))
    if x is None:
        l1 = plt.plot(y, label=label, linewidth=2)
    else:
        l1 = plt.plot(x, y, label=label, linewidth=2)
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.savefig(plot_path + f"/{title}.png")
