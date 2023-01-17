# # Solving a time-dependent problem

#
# This notebook shows how to solve a transient problem using DOLFINx, namely the diffusion problem,
# the simplest extension of the Poisson problem.
#
# The strong form of our problem reads:
# $$
# \begin{align*}
# \frac{\partial T(\boldsymbol{x}, t)}{\partial t} - \nabla \cdot (\alpha  \nabla T(\boldsymbol{x}, t)) &= f(\boldsymbol{x}, t) & & \text{in } \, \Omega, \\
# T(\boldsymbol{x}, t) &= T_D(\boldsymbol{x}, t) & &\text{on} \,\partial\Omega_\text{D}, \\
# \alpha\frac{\partial T}{\partial n} &= 0 & &\text{on} \, \partial\Omega_\text{N} \\
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
#   \frac{T_{n} - T_{n-1}}{\Delta t} - \nabla \cdot (\alpha  \nabla T_{n}) = f_{n}
# \end{align*}
# $$
# Reordering the last equation equation so that only unknows at time step n appear in the
# left-hand side:
# $$
# \begin{align*}
#   T_{n} - \Delta t \nabla \, \cdot (\alpha  \nabla T_{n}) = \Delta t f_{n} + T_{n-1}
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
# regular test function $v$ and applying integration by parts. At time-step $n$ the weak
# reads: Find $u \in V(\Omega)$ such that
#
# $$
# \begin{align*}
#   \int_{\Omega} T_n v \,dx + \Delta t \int_{\Omega}{ \alpha \nabla T_n \cdot \nabla v}\,dx = \Delta t \int_{\Omega} f \cdot v \, dx + \int_{\Omega} T_0 \cdot v \, dx
# \end{align*}
# $$
# For this problem, the natural choice is a space of continuous functions, that are element-wise polynomials of degree $k$
#
# $$
# V(\mathcal{T}_h) = V_h = \{v \in H ^ 1(\Omega), ~v | _E \in P_k(E) \, \forall E \in \mathcal{T}_h\}
# $$

# ## Test problem - Manufactured Solution
# We construct a test problem for which we can easily check the answer. We first define the exact solution by
#
# $$
# \begin{align*}
#   T(\boldsymbol{x}, t) = c_0 x_0^2 + c_1 x_0 + c_3t
# \end{align*}
# $$
# and then we insert the exact solution into the strong form of our pde.
# With simple manipulations we realize that the source should be:
# $$
# \begin{align*}
#   f = c_4 - 2c_0
# \end{align*}
# $$
# Also at the boundaries we have:
# $$
# \begin{align*}
#   T_D(\boldsymbol{x}, t) = c_0 x_0^2 + c_1 x_0 + c_2t
# \end{align*}
# $$

# ## Implementation

# The first step is to import all modules and packages we are going to use:

# +
# Import dolfinx and ufl modules
from dolfinx import geometry
from dolfinx import fem
from dolfinx import io
from dolfinx import la
from dolfinx.mesh import (CellType, create_unit_square, locate_entities, exterior_facet_indices)
from ufl import TestFunction, TrialFunction, dx, inner, grad

# Auxiliary packages
from petsc4py import PETSc
from mpi4py import MPI
import numpy

# Convenience functions for plotting on jupyter notebooks
from utils import plot_mesh, plot_function, create_gif
import IPython
# -

# ### Creating a distributed computational domain (mesh)

# To create a simple computational domain in DOLFINx, we use the mesh generation utilities in `dolfinx.mesh`.
# In this module, we have the tools to build rectangles of triangular or quadrilateral elements and boxes
# of tetrahedral or hexahedral elements. We start by creating a unit square:

# +
comm = MPI.COMM_WORLD
mesh = create_unit_square(comm, 10, 10, CellType.quadrilateral)
# -


# ### Visualizing mesh
# We have provided a few convenience functions for plotting on jupyter notebooks.

# +
plot_mesh(mesh, filename="mesh.html")
IPython.display.HTML(filename="mesh.html")
# -

# ### Handling time-dependent functions expressions
#
# First we define the constants we use in our program:

# +
c0 = fem.Constant(mesh, 1.0)
c1 = fem.Constant(mesh, 2.0)
c2 = fem.Constant(mesh, 0.5)
dt = fem.Constant(mesh, 0.1)
# -

# We can now define the source term as a Constant:
# $$
# \begin{align*}
#   f = c_2 - 2c_0
# \end{align*}
# $$

# +
f = c2 - 2*c0
print(type(f))
# -

# For $T$, $T_0$ and $T_D$ we first define an expression and then we interpolate
# it into the corresponding function space.

# +
def expression(t):
    return lambda x: c0*x[0]**2 + c1*x[0] + c2*t

V = fem.FunctionSpace(mesh, ("Lagrange", 1))
T0 = fem.Function(V)
T0.interpolate(expression(t=0))

# plot T0 over Omega
plot_function(T0, filename="T0.html")
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
#   a(T, v) = \int_{\Omega} T v \,dx + \alpha \Delta t \int_{\Omega}{\nabla T \cdot \nabla v}\,dx
# \end{align*}
# $$

# +
alpha = fem.Constant(mesh, 1.0)
a = inner(T, v) * dx + dt * alpha * inner(grad(T), grad(v)) * dx
a = fem.form(a)  # JIT compilation
# -

# $$
# \begin{align*}
#   L(v) = \Delta t \int_{\Omega} f \cdot v \, dx + \int_{\Omega} T_0 \cdot v \, dx
# \end{align*}
# $$

# +
L = dt * inner(f, v) * dx + inner(T0, v) * dx
# -

# To give the user freedom to set boundary conditions on single degrees of freedom,
# the function `dolfinx.fem.dirichletbc` takes in the list of degrees of freedom(DOFs) as input.
# The DOFs on the boundary can be obtained in many ways: DOLFINx supplies a few convenience functions,
# such as `dolfinx.fem.locate_dofs_topological` and `dolfinx.fem.locate_dofs_geometrical`.
# DOLFINx also has convenience functions to obtain a list of all boundary facets.

# +
# Create Dirichlet function
T_D = fem.Function(V)
T_D.interpolate(expression(0))

# Define Dirichlet bc
tdim = mesh.topology.dim
mesh.topology.create_connectivity(tdim - 1, tdim)
bndry_facets = exterior_facet_indices(mesh.topology)
bndry_dofs = fem.locate_dofs_topological(V, tdim - 1, bndry_facets)
bcs = [fem.dirichletbc(T_D, bndry_dofs)]
# -

# ### Setting up a time dependent solver
#
# As the left hand side of our problem(the matrix) is time independent, we would like avoid re-assembling it at every time step.
# We assemble the matrix once outside the temporal loop.

# +
A = fem.petsc.assemble_matrix(a, bcs=bcs)
A.assemble()
# -

# Next, we can generate the integration kernel for the right hand side(RHS), and create the RHS vector `b` that we will
# assembled into at each time step.

# +
b = fem.Function(V)
L = fem.form(L)  # JIT compilation
# -

# We next create the PETSc KSP(Krylov subspace method) solver, and set it to solve using an
# [algebraic multigrid method](https: // hypre.readthedocs.io/en/latest/solvers-boomeramg.html).

# +
# Define Solver
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
plotter = create_gif(T, "T.gif", [0, 2])
plotter.write_frame()
# -


# ## Solving a time dependent problem
#
# We are now ready to solve the time dependent problem. At each time step, we need to:
# 1. Update the time dependent boundary condition and source
# 2. Reassemble the right hand side vector `b`
# 3. Apply boundary conditions to `b`
# 4. Solve linear problem `AT = b`
# 5. Update current solution, `T0 = T`

# +
t = 0
t_max = 10*dt.value
while t < t_max:
    t += dt.value
    print(f"t = {t:.2f} s")

    # Update boundary condition
    T_D.interpolate(expression(t))

    # Assemble RHS
    b.x.array[:] = 0
    fem.petsc.assemble_vector(b.vector, L)

    # Apply boundary condition
    fem.petsc.apply_lifting(b.vector, [a], [bcs])
    b.x.scatter_reverse(la.ScatterMode.add)
    fem.petsc.set_bc(b.vector, bcs)

    # Solve linear problem
    T.x.array[:] = 0
    solver.solve(b.vector, T.vector)
    T.x.scatter_forward()

    # Update un
    T0.x.array[:] = T.x.array

    # Update plotter
    plotter.update_scalars(T.x.array, render=False)
    plotter.write_frame()


IPython.display.Image(filename="T.gif")
# -

# What's the temperature at an arbitrary point p?

# +
# Given an arbitrary point "p"
p = numpy.array([0.5, 0.5, 0.0], dtype=numpy.float64)

# We first compute the cells that it belongs to
bb_tree = geometry.BoundingBoxTree(mesh, mesh.topology.dim)
cell_candidates = geometry.compute_collisions(bb_tree, p)
cells = geometry.compute_colliding_cells(mesh, cell_candidates, p)

# Given a list of cells it's easy and "efficient" to compute T(p)
values = T.eval(p, cells[0])
print(f"Temperature at point {p} is {values[0]}")
# -

# **TASK 1**: compute the L2 and H1 error norms

# +
eh = T_D - T
M = fem.form(eh**2 * dx)
error_norm =fem.assemble_scalar(M)
print(f"L2 Error norm: {error_norm}")
# -

# **TASK 2**: Change the cell type and polynomial degree.
# Increase and decrease the time step dt.
# Why doesn't the norm of error change?


# # Homework 1
# In this homework we are going to tackle a more physically realistic problem.
# Given the temperature outside a room, we shall compute the temperature inside
# the room, and find the appropriate insulation material. 
#
# Without internal sources the strong form of our problem reads:
# $$
# \begin{align*}
# \frac{\partial T}{\partial t} &= \nabla \cdot (\alpha  \nabla T) & & \text{in } \, \Omega, \\
# T(\boldsymbol{x}, t) &= T_D = T_R + T_a \cdot sin(\omega t) & & \,\partial\Omega_\text{D}, \\
# T(\boldsymbol{x}, t=0) &= T_0 & & \text{in } \, \Omega,
# \end{align*}
# $$
# Where $\alpha$ is the thermal diffusivity of the medium:
# $$
# \begin{align*}
# \alpha = \frac{k}{\rho C_p}
# \end{align*}
# $$
# $k$ is the thermal conductivity, $\rho$ is the density and $C_p$ is the is specific heat capacity.
# Examples of thermal diffusivity for four different materials:
# $$\alpha_{air} = 19 \cdot 10^{-6} \, m^2/s$$
# $$\alpha_{brick} = 0.27 \cdot 10^{-6} \, m^2/s$$
# $$\alpha_{wood} = 0.082 \cdot 10^{-6} \, m^2/s$$
# $$\alpha_{steel} = 11.72 \cdot 10^{-6} \, m^2/s$$
#
# Given a 1 squared meter room, 0.1 m thick walls, and with outside 
# temperature given by $T_D$ which material should be used for the wall 
# such that the temperature at the center of the room is never below $10^oC$?

# **TASK 1**: Complete the following code with the thermal diffusivity of the air and
# the wall and visualize the material distribution:

# +
def wall(x):
    vertical = (x[0] <= 0.1) | (x[0] >= 0.9)
    horizontal = (x[1] <= 0.1) | (x[1] >= 0.9)
    return horizontal | vertical


comm = MPI.COMM_WORLD
mesh = create_unit_square(comm, 20, 20, CellType.quadrilateral)

cells = locate_entities(mesh, mesh.topology.dim, wall)
DG = fem.FunctionSpace(mesh, ("DG", 0))

# Thermal diffusivity
alpha = fem.Function(DG)
# alpha.x.array[:] = alpha_air
# alpha.x.array[cells] = alpha_wall

# Solution - 
alpha.x.array[:] = 19e-6
alpha.x.array[cells] = 0.082e-6 # wood

plot_mesh(mesh, cell_values=alpha, filename="alpha.html")
IPython.display.HTML(filename="alpha.html")
# -


# **Task 2**: create a python function that defines the outside temperature
# $$T_{out} = -5 + 15 \cdot sin(\omega t)$$
# where $\omega = \pi/(24\cdot60\cdot60)$.

# +
period = 60*60*24 # seconds in a day

# Task: define temperature outside temperature and plot it for 
def outside_temperature(t):
    omega = numpy.pi/period
    t = t % period
    temperature = 15 * numpy.sin(omega*t) - 5
    print(f"Outside temperature = {temperature}")
    return lambda x: numpy.full_like(x[0], temperature)

# Task: Plot the hourly temperature during a day:?
temp = numpy.zeros(24)
for i in range(temp.size):
    t = i*0.5*60*60
    temp[i] = outside_temperature(t)([0,0])

import matplotlib.pyplot as plt

plt.plot(temp)
plt.savefig("temperature.png")
IPython.display.Image(filename="temperature.png")
# -

# **Task 3** Modify the following function to return the hourly variation 
# of temperature inside the room.

# +
def thermal_solver(alpha):
    mesh = alpha.function_space.mesh

    # Create function space
    V = fem.FunctionSpace(mesh, ("Lagrange", 1))
    
    # Define trial and test functions
    T = TrialFunction(V)
    v = TestFunction(V)

    # Initial condition
    T0 = fem.Function(V)
    T0.interpolate(lambda x: numpy.full_like(x[0], 12))

    # Time step (15 min - 900 seconds)
    dt = fem.Constant(mesh, 900.0)

    # Bilinear form
    a = inner(T, v) * dx + dt * alpha * inner(grad(T), grad(v)) * dx
    a = fem.form(a)

    # Linear form
    L = inner(T0, v) * dx
    L = fem.form(L)  # JIT compilation


    # Define dirichlet bc at t=0
    T_D = fem.Function(V)
    T_D.interpolate(outside_temperature(0))
    
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    bndry_facets = exterior_facet_indices(mesh.topology)
    bndry_dofs = fem.locate_dofs_topological(V, tdim - 1, bndry_facets)
    bcs = [fem.dirichletbc(T_D, bndry_dofs)]

    # Assemble matrix A
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
    T.interpolate(T0)
    plotter = create_gif(T, "room.gif", clim=[-5, 15])
    plotter.write_frame()

    # Check against standard table value
    # Create bounding box for function evaluation
    bb_tree = geometry.BoundingBoxTree(mesh, mesh.topology.dim)
    p = numpy.array([0.5, 0.5, 0.0], dtype=numpy.float64)
    cell_candidates = geometry.compute_collisions(bb_tree, p)
    cells = geometry.compute_colliding_cells(mesh, cell_candidates, p)

    value = T.eval(p, cells[0])
    print(f"Inside temperature {value}")


    t = 0
    t_max = period * 1

    # Array to store temperature
    temperature = numpy.zeros(int(t_max/dt.value))

    while t < t_max:
        t += dt.value
        print(f"t = {t/3600.0} h")

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

        value = T.eval(p, cells[0])
        print(f"Inside temperature {value}")

        # Update un
        T0.x.array[:] = T.x.array

        # Update plotter
        plotter.update_scalars(T.x.array, render=False)
        plotter.write_frame()
    
    return temperature
# -

# **Task 4**: Which material should we use for insulation?
# Plot the T vs time for different types of material for the wall.

# +
alpha.x.array[cells] = 0.082e-6 # wood
temp = thermal_solver(alpha)
# -