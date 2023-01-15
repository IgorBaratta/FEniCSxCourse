# +
# @title Install dolfinx and dependencies
try:
from matplotlib import pyplot
from dolfinx import plot
import pyvista
import gmsh
except ImportError:
    !wget "https://github.com/fem-on-colab/fem-on-colab.github.io/raw/7f220b6/releases/gmsh-install.sh" - O "/tmp/gmsh-install.sh" & & bash "/tmp/gmsh-install.sh"
    import gmsh

try:
    import dolfinx
except ImportError:
    !wget "https://github.com/fem-on-colab/fem-on-colab.github.io/raw/7f220b6/releases/fenicsx-install-real.sh" - O "/tmp/fenicsx-install.sh" & & bash "/tmp/fenicsx-install.sh"
    import dolfinx

try:
    import pyvista
except ImportError:
    !pip install - q piglet pyvirtualdisplay ipyvtklink pyvista panel
    !apt-get - qq install xvfb
    import pyvista
# -

# # Solving a time-dependent problem

#
# This notebook shows how to solve a transient problem using DOLFINx, namely the diffusion problem,
# the simplest extension of the Poisson problem.
#
# The strong form of our problem reads:
# $$
# \begin{align*}
# \frac{\partial T(\boldsymbol{x}, t)}{\partial t} - \nabla \cdot (\mu  \nabla T(\boldsymbol{x}, t)) &= f(\boldsymbol{x}, t) & & \text{in } \, \Omega, \\
# T(\boldsymbol{x}, t) &= T_D(\boldsymbol{x}, t) & &\text{on} \,\partial\Omega_\text{D}, \\
# \mu\frac{\partial T}{\partial n} &= 0 & &\text{on} \, \partial\Omega_\text{N} \\
# T(\boldsymbol{x}, t=0) &= T_i(\boldsymbol{x}) & & \text{in } \, \Omega,
# \end{align*}
# $$
#
# Where $T$, the temperature distribution, varies with space and time $T(\boldsymbol{x}, t)$.
# $T_D$ is a prescribed function at the boundary $\partial\Omega_\text{D}$.
# And $T_i$ is the initial temperature distribution.
#
# To solve time-dependent PDEs using the finite element method, we first discretize the time derivative
# using a finite difference approximation, yielding a recursive series of stationary problems,
# and then we convert each stationary problem into a variational problem.
#


# ## Time discretization
# A backward Euler scheme can be used to approximate the time derivative.
# $$
# \begin{align*}
#   \frac{T_{n+1} - T_{n}}{\Delta t} - \nabla \cdot (\mu  \nabla T_{n+1}) = f_{n+1}
# \end{align*}
# $$
# Reordering the last equation equation so that only $T_{n+1}$ appears in the left-hand
# side:
# $$
# \begin{align*}
#   T_{n+1} - \Delta t \nabla \, \cdot (\mu  \nabla T_{n+1}) = \Delta t f_{n+1} + T_{n}
# \end{align*}
# $$
#

# ## Time-stepping algorithm
#       Compute T_0 as interpolation of a given function T_i(\boldsymbol{x}) 
#       Define the bilinear a(T,v) and linear L(v) forms
#       Assemble the matrix A from the bilinear form a
#       t = 0
#       while t < t_max:
#           Assemble the vector b from the linear form L
#           Apply time dependent boundary conditions
#           Solve the linear system AT=b
#           Update current solution T_n = T



# ## Variational formulation
# As usual we find the weak by multiplying our semi-discrete equation by a sufficiently 
# regular test function $v$ and applying integration by parts


#
# Relevant DOLFINx modules:
# - `dolfinx.mesh`: Classes and functions related to the computational domain
# - `dolfinx.fem`: Finite element method functionality
# - `dolfinx.io`: Input/Output(read/write) functionality
# - `dolfinx.plot`: Convenience functions for exporting plotting data
# - `dolfinx.la`: Functions related to linear algebra structures(matrices/vectors)

# +
from petsc4py import PETSc
import numpy as np
from ufl import (TestFunction, SpatialCoordinate, TrialFunction,
                 as_vector, dx, grad, inner, system)
import IPython
from dolfinx import mesh, fem, la
from mpi4py import MPI
# -

# ## Creating a distributed computational domain (mesh)

# To create a simple computational domain in DOLFINx, we use the mesh generation utilities in `dolfinx.mesh`.
# In this module, we have the tools to build rectangles of triangular or quadrilateral elements and boxes
# of tetrahedral or hexahedral elements. We start by creating a rectangle spanning $[0, 0]\times[10, 3]$,
# with 100 and 20 elements in each direction respectively.

# +
length, height = 10, 3
Nx, Ny = 40, 30
extent = [[0., 0.], [length, height]]
comm = MPI.COMM_WORLD
domain = mesh.create_rectangle(
    comm, extent, [Nx, Ny], mesh.CellType.quadrilateral)
# -

# ## Visualizing mesh
# +


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
# -


# +
plot_mesh(domain, "mesh.html")
IPython.display.HTML(filename="mesh.html")
# -


# # Setting up a variational problem

# We will solve the heat equation, with a backward Euler time stepping scheme, ie
#
# $$
# \begin{align*}
# \frac{u_{n+1}-u_n}{\Delta t} - \nabla \cdot (\mu  \nabla u_{n+1}) &= f(x, t_{n+1}) & & \text{in } \Omega, \\
# u &= u_D(x, t_{n+1}) & &\text{on} \partial\Omega_\text{D}, \\
# \mu\frac{\partial u_{n+1}}{\partial n} &= 0 & &\text{on} \partial\Omega_\text{N},
# \end{align*}
# $$
# with $u_D = y\cos(0.25t)$, $f = 0$. For this example, we take $\Omega$ to be rectangle defined above, $\Omega_\text{D}$
# if the left-hand edge of the rectangle, and $\Omega_\text{N}$ is the remaining three edges of the rectangle.
#
# We start by defining the function space, the corresponding test and trial functions, as well as material and temporal parameters.
# Note that we use explicit imports from UFL to create the test and trial functions, to avoid confusion as to where they originate from .
#
# DOLFINx and UFL support both real and complex valued functions. However, to be able to use the PETSc linear algebra backend,
# which only supports a single floating type at compilation, we need to use appropriate scalar types in our variational form.
# This ensures that we generate consistent matrices and vectors.

# +
V = fem.FunctionSpace(domain, ("Lagrange", 1))
u = TrialFunction(V)
v = TestFunction(V)
un = fem.Function(V)
f = fem.Constant(domain, 0.0)
mu = fem.Constant(domain, 2.3)
dt = fem.Constant(domain, 0.05)
# -

# The variational form can be written in UFL syntax:

# +
F = inner(u - un, v) * dx + dt * mu * inner(grad(u), grad(v)) * dx
F -= dt * inner(f, v) * dx
(a, L) = system(F)
# -

# ## Creating Dirichlet boundary conditions

# ### Creating a time dependent boundary condition

# There are many ways of creating boundary conditions. In this example, we will create
# function $u_\text{D}(x, t)$ dependent on both space and time. To do this, we define a
# function that takes a 2-dimensional array `x`.  Each column of `x` corresponds to an input
# coordinate $(x, y, z)$ and this function operates directly on the columns of `x`.
#

# +


def uD_function(t):
    return lambda x: x[1] * np.cos(0.25 * t)


uD = fem.Function(V)
t = 0
uD.interpolate(uD_function(t))

filename = uD.name + ".html"
plotter = plot_function(uD, filename)
IPython.display.HTML(filename=filename)
# -

# To give the user freedom to set boundary conditions on single degrees of freedom,
# the function `dolfinx.fem.dirichletbc` takes in the list of degrees of freedom(DOFs) as input.
# The DOFs on the boundary can be obtained in many ways: DOLFINx supplies a few convenience functions,
# such as `dolfinx.fem.locate_dofs_topological` and `dolfinx.fem.locate_dofs_geometrical`.
# Locating dofs topologically is generally advised, as certain finite elements have DOFs that do not have a
# geometrical coordinates associated with them(eg Nédélec and Raviart--Thomas).
# DOLFINx also has convenience functions to obtain a list of all boundary facets.

# +
tdim = domain.topology.dim
fdim = tdim - 1
bndry_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.isclose(x[0], length))

bndry_dofs = fem.locate_dofs_topological(V, tdim - 1, bndry_facets)
bcs = [fem.dirichletbc(uD, bndry_dofs)]
# -

# # Setting up a time dependent solver

# As the left hand side of our problem(the matrix) is time independent, we would like avoid re-assembling it at every time step.
# DOLFINx gives the user more control over assembly so that this can be done. We assemble the matrix once outside the temporal loop.

# +
compiled_a = fem.form(a)
A = fem.petsc.assemble_matrix(compiled_a, bcs=bcs)
A.assemble()
# -

# Next, we can generate the integration kernel for the right hand side(RHS), and create the RHS vector `b` that we will
# assemble into at each time step.

# +
compiled_L = fem.form(L)
b = fem.Function(V)
# -

# We next create the PETSc KSP(Krylov subspace method) solver, and set it to solve using an
# [algebraic multigrid method](https: // hypre.readthedocs.io/en/latest/solvers-boomeramg.html).

# +
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.CG)
pc = solver.getPC()
pc.setType(PETSc.PC.Type.HYPRE)
pc.setHYPREType("boomeramg")
# -

# ## Plotting a time dependent problem

# As we are solving a time dependent problem, we would like to create a time dependent animation of the solution.
# We do this by using [pyvista](https: // docs.pyvista.org /), which uses VTK structures for plotting.
# In DOLFINx, we have the convenience function `dolfinx.plot.create_vtk_mesh` that can create meshes compatible
# with VTK formatting, based on meshes of(discontinuous) Lagrange function spaces.

# +
uh = fem.Function(V)
plotter = plot_function(uh, uh.name + ".gif")
plotter.write_frame()
# -

# ## Solving a time dependent problem
#
# We are now ready to solve the time dependent problem. At each time step, we need to:
# 1. Update the time dependent boundary condition and source
# 2. Reassemble the right hand side vector `b`
# 3. Apply boundary conditions to `b`
# 4. Solve linear problem `Au = b`
# 5. Update previous time step, `un = u`

# +
T = 3.14
while t < T:
    t += dt.value
    print(f"t = {t}")

    # Update boundary condition
    uD.interpolate(uD_function(t))

    # Assemble RHS
    b.x.array[:] = 0
    fem.petsc.assemble_vector(b.vector, compiled_L)

    # Apply boundary condition
    fem.petsc.apply_lifting(b.vector, [compiled_a], [bcs])
    b.x.scatter_reverse(la.ScatterMode.add)
    fem.petsc.set_bc(b.vector, bcs)

    # Solve linear problem
    solver.solve(b.vector, uh.vector)
    uh.x.scatter_forward()

    # Update un
    un.x.array[:] = uh.x.array

    # Update plotter
    plotter.update_scalars(uh.x.array, render=False)
    plotter.write_frame()

IPython.display.Image(filename=uh.name + ".gif")
# -


# ## Post-processing without projections

# In legacy dolfin, the only way of post-processing a `ufl`- expression over the domain, would be by using a projection.
# This would not be scalable in most cases. Therefore, we have introduced `dolfinx.fem.Expression`, which can be used to
# evaluate a `ufl`- expression at any given(reference) point in any cell(local to process). Let us consider
#
# $$(y, x) \cdot(\nabla u)$$

# +
x = SpatialCoordinate(domain)
x_grad = inner(as_vector((x[1], x[0])), grad(uh))

W = fem.FunctionSpace(domain, ("DQ", 1))

expr = fem.Expression(x_grad, W.element.interpolation_points())
w = fem.Function(W)
w.interpolate(expr)

plot_function(w, "w.html")
IPython.display.HTML(filename="w.html")
# -
