from matplotlib import pyplot
from dolfinx import plot
import pyvista
from scipy.special import jv, hankel1
import numpy as np

pyvista.set_plot_theme("paraview")


def plot_mesh(mesh, cell_values=None, filename="file.html"):
    pyvista.start_xvfb()
    grid = pyvista.UnstructuredGrid(*plot.create_vtk_mesh(mesh))
    plotter = pyvista.Plotter(notebook=True)

    if cell_values is not None:
        min_ = cell_values.x.array.min()
        max_ = cell_values.x.array.max()
        grid.cell_data["cell_values"] = cell_values.x.array.real
        viridis = pyplot.cm.get_cmap("viridis", 25)
        plotter.add_mesh(grid, cmap=viridis,
                         show_edges=True, clim=[min_, max_])
    else:
        plotter.add_mesh(grid, show_edges=True)

    plotter.camera.zoom(2.0)
    plotter.view_xy()
    plotter.export_html(filename, backend="pythreejs")
    plotter.close()


def penetrable_circle(k0, k1, rad, plot_grid):
    # Compute 
    points = np.vstack((plot_grid[0].ravel(), plot_grid[1].ravel()))
    fem_xx = points[0, :]
    fem_xy = points[1, :]
    r = np.sqrt(fem_xx * fem_xx + fem_xy * fem_xy)
    theta = np.arctan2(fem_xy, fem_xx)
    npts = np.size(fem_xx, 0)
    a = rad

    n_terms = np.max([100, np.int(55 + (k0 * a)**1.01)])

    Nx = plot_grid.shape[1]
    Ny = plot_grid.shape[2]

    u_inc = np.exp(1j * k0 * fem_xx)
    n_int = np.where(r < a)
    n_ext = np.where(r >= a)
    u_inc[n_int] = 0.0
    u_plot = u_inc.reshape(Nx, Ny)

    u_int = np.zeros(npts, dtype=np.complex128)
    u_ext = np.zeros(npts, dtype=np.complex128)
    n = np.arange(-n_terms, n_terms+1)

    bessel_k0 = jv(n, k0 * rad)
    bessel_k1 = jv(n, k1 * rad)
    hankel_k0 = hankel1(n, k0 * rad)

    bessel_deriv_k0 = jv(n-1, k0 * rad) - n/(k0 * rad) * jv(n, k0 * rad)
    bessel_deriv_k1 = jv(n-1, k1 * rad) - n/(k1 * rad) * jv(n, k1 * rad)

    hankel_deriv_k0 = n/(k0 * rad) * hankel_k0 - hankel1(n+1, k0 * rad)

    a_n = (1j**n) * (k1 * bessel_deriv_k1 * bessel_k0 -
                     k0 * bessel_k1 * bessel_deriv_k0) / \
        (k0 * hankel_deriv_k0 * bessel_k1 -
         k1 * bessel_deriv_k1 * hankel_k0)
    b_n = (a_n * hankel_k0 + (1j**n) * bessel_k0) / bessel_k1

    # compute u_ext += a_n * hankel1(n, k0 * r) * np.exp(1j * n_ * theta)
    k0r = np.tile(k0*r, (a_n.size, 1))
    n_mat = np.repeat(n, r.size).reshape((a_n.size, r.size))
    M = np.diag(a_n)@hankel1(n_mat, k0r)
    u = np.exp(1j * np.outer(n, theta))
    u_ext += np.sum(M*u, axis=0)

    # u_int += b_n * jv(n_, k1 * r) * np.exp(1j * n_ * theta)
    k1r = np.tile(k1*r, (b_n.size, 1))
    n_mat = np.repeat(n, r.size).reshape((a_n.size, r.size))
    M = np.diag(b_n)@jv(n_mat, k1r)
    u = np.exp(1j * np.outer(n, theta))
    u_int += np.sum(M*u, axis=0)

    u_int[n_ext] = 0.0
    u_ext[n_int] = 0.0
    u_sc = u_int + u_ext
    u_scat = u_sc.reshape(Nx, Ny)
    u_tot = u_scat + u_plot

    return u_tot
