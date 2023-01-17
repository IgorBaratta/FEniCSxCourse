from matplotlib import pyplot
from dolfinx import plot
import pyvista
import numpy

pyvista.set_plot_theme("paraview")


def plot_mesh(mesh, cell_values=None, filename="file.html"):
    pyvista.start_xvfb()
    grid = pyvista.UnstructuredGrid(*plot.create_vtk_mesh(mesh))
    plotter = pyvista.Plotter(notebook=True)

    if cell_values is not None:
        min_ = cell_values.x.array.min()
        max_ = cell_values.x.array.max()
        grid.cell_data["cell_values"] = cell_values.x.array
        viridis = pyplot.cm.get_cmap("viridis", 25)
        plotter.add_mesh(grid, cmap=viridis,
                         show_edges=True, clim=[min_, max_])
    else:
        plotter.add_mesh(grid, show_edges=True)

    plotter.camera.zoom(2.0)
    plotter.view_xy()
    plotter.export_html(filename, backend="pythreejs")
    plotter.close()


def plot_function(uh, filename):
    pyvista.start_xvfb(0.5)
    plotter = pyvista.Plotter(notebook=False, off_screen=True)
    V = uh.function_space
    topology, cells, geometry = plot.create_vtk_mesh(V.mesh)
    grid = pyvista.UnstructuredGrid(topology, cells, geometry)
    grid.point_data["uh"] = numpy.abs(uh.x.array)
    viridis = pyplot.cm.get_cmap("viridis", 25)

    min_ = uh.x.array.min()
    max_ = uh.x.array.max()

    plotter.add_mesh(grid, show_edges=True, lighting=False,
                     cmap=viridis, clim=[min_, max_])
    plotter.camera.zoom(2.0)
    plotter.view_xy()
    plotter.export_html(filename, backend="pythreejs")
    plotter.close()


def create_gif(uh, filename, clim):
    pyvista.start_xvfb(0.5)
    plotter = pyvista.Plotter(notebook=False, off_screen=True)
    plotter.open_gif(filename)
    V = uh.function_space
    topology, cells, geometry = plot.create_vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cells, geometry)
    grid.point_data["uh"] = uh.x.array
    viridis = pyplot.cm.get_cmap("viridis", 25)

    plotter.add_mesh(grid, show_edges=True, lighting=False,
                     cmap=viridis, clim=clim)
    plotter.camera.zoom(2.0)
    plotter.view_xy()

    return plotter
