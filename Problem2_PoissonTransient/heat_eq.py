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


# ## Test problem
# We construct a test problem for which we can easily check the answer. We first define the exact solution by
#
# $$
# \begin{align*}
#   T(\boldsymbol{x}, t) = c_0 x_0^2 - c_1 x_0 + c_2 x_1^2 - c_3 x_1 + c_4t
# \end{align*}
# $$
#
# $$
# \begin{align*}
#   f = c_4 - 2(c_0 + c_3)
# \end{align*}
# $$
#
# $$
# \begin{align*}
#   T(\boldsymbol{x}, 0) = c_0 x_0^2 - c_1 x_0 + c_2 x_1^2 - c_3 x_1
# \end{align*}
# $$
#
# $$
# \begin{align*}
#   T_D(\boldsymbol{x}, t) = c_0 x_0^2 - c_1 x_0 + c_2 x_1^2 - c_3 x_1 + c_4t
# \end{align*}
# $$

# ### Creating a distributed computational domain (mesh)

# To create a simple computational domain in DOLFINx, we use the mesh generation utilities in `dolfinx.mesh`.
# In this module, we have the tools to build rectangles of triangular or quadrilateral elements and boxes
# of tetrahedral or hexahedral elements. We start by creating a unit square:

# +
from dolfinx import io
from dolfinx.mesh import locate_entities
import numpy
from petsc4py import PETSc
from dolfinx.mesh import exterior_facet_indices
from ufl import inner, grad, dx
from ufl import TestFunction, TrialFunction
from matplotlib import pyplot
from dolfinx import fem
import pyvista
import IPython
from dolfinx import plot
from dolfinx.mesh import create_unit_square
from mpi4py import MPI

comm = MPI.COMM_WORLD
mesh = create_unit_square(comm, 10, 10)
# -

# ### Visualizing mesh

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


plot_mesh(mesh, "mesh.html")
IPython.display.HTML(filename="mesh.html")
# -

# ### Handling time-dependent functions expressions

# +

# Define our 5 constants c0, c1, ..., c4
c0 = fem.Constant(mesh, -3.0)
c1 = fem.Constant(mesh, 2.0)
c2 = fem.Constant(mesh, -1.0)
c3 = fem.Constant(mesh, 2.0)
c4 = fem.Constant(mesh, 0.3)
dt = fem.Constant(mesh, 0.1)
# -

# $$
# \begin{align*}
#   f = c_4 - 2(c_0 + c_3)
# \end{align*}
# $$

# +
f = c4 - 2*(c0 + c3)
# -

# +


def exact_solution(t):
    return lambda x: c0*x[0]**2 + c1*x[0] + c2*x[1]**2 + c3*x[1] + c4*t


V = fem.FunctionSpace(mesh, ("Lagrange", 1))
T0 = fem.Function(V)
T0.interpolate(exact_solution(0))
# -


# ### Visualizing a function

# +


def plot_function(uh, filename):
    pyvista.start_xvfb(0.5)
    plotter = pyvista.Plotter(notebook=False, off_screen=True)
    if "gif" in filename:
        plotter.open_gif(filename)

    V = uh.function_space
    topology, cells, geometry = plot.create_vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cells, geometry)
    grid.point_data["uh"] = uh.x.array
    viridis = pyplot.cm.get_cmap("viridis", 25)
    # sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
    #              position_x=0.1, position_y=0.8, width=0.8, height=0.1)
    plotter.add_mesh(grid, show_edges=True, lighting=False, cmap=viridis)
    plotter.camera.zoom(2.0)
    plotter.view_xy()

    if "html" in filename:
        plotter.export_html(filename, backend="pythreejs")

    return plotter


plot_function(T0, "T0.html")
IPython.display.HTML(filename="T0.html")
# -


# ## Setting up a variational problem
# +
V = fem.FunctionSpace(mesh, ("Lagrange", 1))
T = TrialFunction(V)
v = TestFunction(V)
T0 = fem.Function(V)
# -


# The variational form can be written in UFL syntax:

# $$
# \begin{align*}
#   a(T, v) = \int_{\Omega} T v \,dx + \mu \Delta t \int_{\Omega}{\nabla T \cdot \nabla v}\,dx
# \end{align*}
# $$


# +
mu = fem.Constant(mesh, 1.0)
a = inner(T, v) * dx + dt * mu * inner(grad(T), grad(v)) * dx
a = fem.form(a)
# -

# $$
# \begin{align*}
#   L(v) = \Delta t \int_{\Omega} f * v \, dx + \int_{\Omega} T_0 * v \, dx
# \end{align*}
# $$

# +
L = dt * inner(f, v) * dx + inner(T0, v) * dx
# -


# To give the user freedom to set boundary conditions on single degrees of freedom,
# the function `dolfinx.fem.dirichletbc` takes in the list of degrees of freedom(DOFs) as input.
# The DOFs on the boundary can be obtained in many ways: DOLFINx supplies a few convenience functions,
# such as `dolfinx.fem.locate_dofs_topological` and `dolfinx.fem.locate_dofs_geometrical`.
# Locating dofs topologically is generally advised, as certain finite elements have DOFs that do not have a
# geometrical coordinates associated with them(eg Nédélec and Raviart--Thomas).
# DOLFINx also has convenience functions to obtain a list of all boundary facets.

# +

T_D = fem.Function(V)
T_D.interpolate(exact_solution(0))

tdim = mesh.topology.dim
mesh.topology.create_connectivity(tdim - 1, tdim)
bndry_facets = exterior_facet_indices(mesh.topology)
bndry_dofs = fem.locate_dofs_topological(V, tdim - 1, bndry_facets)
bcs = [fem.dirichletbc(T_D, bndry_dofs)]
# -

# ### Setting up a time dependent solver

# As the left hand side of our problem(the matrix) is time independent, we would like avoid re-assembling it at every time step.
# We assemble the matrix once outside the temporal loop.

# +
A = fem.petsc.assemble_matrix(a, bcs=bcs)
A.assemble()
# -

# Next, we can generate the integration kernel for the right hand side(RHS), and create the RHS vector `b` that we will
# assemble into at each time step.

# +
b = fem.Function(V)
L = fem.form(L)  # JIT compilation
# -


# # We next create the PETSc KSP(Krylov subspace method) solver, and set it to solve using an
# # [algebraic multigrid method](https: // hypre.readthedocs.io/en/latest/solvers-boomeramg.html).
# +

solver = PETSc.KSP().create(comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.CG)
pc = solver.getPC()
pc.setType(PETSc.PC.Type.HYPRE)
pc.setHYPREType("boomeramg")
# -


# ### Plotting a time dependent problem

# As we are solving a time dependent problem, we would like to create a time dependent animation of the solution.
# We do this by using [pyvista](https: // docs.pyvista.org /), which uses VTK structures for plotting.
# In DOLFINx, we have the convenience function `dolfinx.plot.create_vtk_mesh` that can create meshes compatible
# with VTK formatting, based on meshes of(discontinuous) Lagrange function spaces.

# +
T = fem.Function(V)
plotter = plot_function(T, "T.gif")
plotter.write_frame()
# -


# ## Solving a time dependent problem
#
# We are now ready to solve the time dependent problem. At each time step, we need to:
# 1. Update the time dependent boundary condition and source
# 2. Reassemble the right hand side vector `b`
# 3. Apply boundary conditions to `b`
# 4. Solve linear problem `AT = b`
# 5. Update previous time step, `T0 = T`

# # +
# from dolfinx.la import ScatterMode

# t = 0
# t_max = 2
# while t < t_max:
#     t += dt.value
#     print(f"t = {t}")

#     # Update boundary condition
#     T_D.interpolate(exact_solution(t))

#     # Assemble RHS
#     b.x.array[:] = 0
#     fem.petsc.assemble_vector(b.vector, L)

#     # Apply boundary condition
#     fem.petsc.apply_lifting(b.vector, [a], [bcs])
#     b.x.scatter_reverse(ScatterMode.add)
#     fem.petsc.set_bc(b.vector, bcs)

#     # Solve linear problem
#     solver.solve(b.vector, T.vector)
#     T.x.scatter_forward()

#     # Update un
#     T0.x.array[:] = T.x.array

#     # Update plotter
#     plotter.update_scalars(T.x.array, render=False)
#     plotter.write_frame()

# IPython.display.Image(filename="T.gif")
# # -


# ## Homework 1


def wall(x):
    bottom = x[0] <= 0.1 + 1e-10
    top = x[0] >= 0.9 - 1e-10
    left = x[1] <= 0.1 + 1e-10
    right = x[1] >= 0.9 - 1e-10
    vertical = numpy.logical_or(bottom, top)
    horizontal = numpy.logical_or(left, right)
    return numpy.logical_or(vertical, horizontal)


cells = locate_entities(mesh, mesh.topology.dim, wall)
DG = fem.FunctionSpace(mesh, ("DG", 0))

# Thermal diffusivity
alpha = fem.Function(DG)
alpha.x.array[:] = 0.0082
alpha.x.array[cells] = 0.0082


with io.XDMFFile(MPI.COMM_WORLD, "inclusions.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(alpha)

period = 3600


def outside_temperature(t):
    omega = numpy.pi/period
    temperature = 25 * numpy.sin(omega*t)
    print(f"Outside temperature = {temperature}")
    return lambda x: numpy.full_like(x[0], temperature)


def thermal_solver(alpha):
    V = fem.FunctionSpace(mesh, ("Lagrange", 1))

    T = TrialFunction(V)
    v = TestFunction(V)
    T0 = fem.Function(V)
    T0.interpolate(lambda x: numpy.full_like(x[0], 0))

    dt = fem.Constant(mesh, 5.0)

    a = inner(T, v) * dx + dt * alpha * inner(grad(T), grad(v)) * dx
    a = fem.form(a)
    
    L = inner(T0, v) * dx
    L = fem.form(L)  # JIT compilation


    T_D = fem.Function(V)
    T_D.interpolate(outside_temperature(0))
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    bndry_facets = exterior_facet_indices(mesh.topology)
    bndry_dofs = fem.locate_dofs_topological(V, tdim - 1, bndry_facets)
    bcs = [fem.dirichletbc(T_D, bndry_dofs)]

    A = fem.petsc.assemble_matrix(a, bcs=bcs)
    A.assemble()

    b = fem.Function(V)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    pc.setHYPREType("boomeramg")

    T = fem.Function(V)
    plotter = plot_function(T, "room.gif")
    plotter.write_frame()


    t = 0
    t_max = 600
    while t < t_max:
        t += dt.value
        print(f"t = {t/3600.0}")

        # Update boundary condition
        T_D.interpolate(outside_temperature(t))

        # Assemble RHS
        b.x.array[:] = 0
        fem.petsc.assemble_vector(b.vector, L)

        # Apply boundary condition
        fem.petsc.apply_lifting(b.vector, [a], [bcs])
        # b.x.scatter_reverse(ScatterMode.add)
        fem.petsc.set_bc(b.vector, bcs)

        # Solve linear problem
        solver.solve(b.vector, T.vector)
        T.x.scatter_forward()

        # Update un
        T0.x.array[:] = T.x.array

        # Update plotter
        plotter.update_scalars(T.x.array, render=False)
        plotter.write_frame()


thermal_solver(alpha)