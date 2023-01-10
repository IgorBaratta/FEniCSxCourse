import pyvista
from dolfinx import plot
from matplotlib import pyplot


def plot_mesh(mesh, filename="file.html"):
    pyvista.start_xvfb()
    grid = pyvista.UnstructuredGrid(*plot.create_vtk_mesh(mesh))
    plotter = pyvista.Plotter(notebook=True)
    plotter.add_mesh(grid, show_edges=True)
    plotter.camera.zoom(2.0)
    plotter.view_xy()
    plotter.export_html(filename, backend="pythreejs")
    plotter.close()


def plot_function(uh, filename):
    # Start virtual framebuffer for plotting
    pyvista.start_xvfb(0.5)
    plotter = pyvista.Plotter(notebook=False, off_screen=True)
    if "gif" in filename:
        plotter.open_gif(filename)

    V = uh.function_space
    topology, cells, geometry = plot.create_vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cells, geometry)
    grid.point_data["uh"] = uh.x.array
    viridis = pyplot.cm.get_cmap("viridis", 25)
    sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
                 position_x=0.1, position_y=0.8, width=0.8, height=0.1)
    plotter.add_mesh(grid, show_edges=True, lighting=False,
                     cmap=viridis, scalar_bar_args=sargs)
    plotter.camera.zoom(2.0)
    plotter.view_xy()

    if "html" in filename:
        plotter.export_html(filename, backend="pythreejs")

    return plotter
