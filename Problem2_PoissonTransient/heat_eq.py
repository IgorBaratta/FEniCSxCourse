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

# ## Creating a distributed computational domain (mesh)

# To create a simple computational domain in DOLFINx, we use the mesh generation utilities in `dolfinx.mesh`.
# In this module, we have the tools to build rectangles of triangular or quadrilateral elements and boxes
# of tetrahedral or hexahedral elements. We start by creating a unit square:

# +
from dolfinx.mesh import create_unit_square
from mpi4py import MPI

comm = MPI.COMM_WORLD
mesh = create_unit_square(comm, 10, 10)
# -
