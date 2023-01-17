# +
try:
  import gmsh
except ImportError:
  !wget "https://github.com/fem-on-colab/fem-on-colab.github.io/raw/7f220b6/releases/gmsh-install.sh" -O "/tmp/gmsh-install.sh" && bash "/tmp/gmsh-install.sh"
  import gmsh

try:
    import dolfinx
except ImportError:
    !wget "https://fem-on-colab.github.io/releases/fenicsx-install-complex.sh" -O "/tmp/fenicsx-install.sh" && bash "/tmp/fenicsx-install.sh"
    import dolfinx

try: 
  import pyvista
except ImportError:
  !pip install -q piglet pyvirtualdisplay ipyvtklink pyvista panel
  !apt-get -qq install xvfb
  import pyvista

!wget "https://raw.githubusercontent.com/IgorBaratta/FEniCSxCourse/main/Problem7_Helmholtz/utils.py"
!wget "https://raw.githubusercontent.com/IgorBaratta/FEniCSxCourse/main/Problem7_Helmholtz/mesh_generation.py"

# -

# # The Helmholtz equation
#
# In this tutorial, we will learn:
# - How to solve PDEs with complex-valued fields,
# - How to import and use high-order meshes from Gmsh,
# - How to use high order discretizations,
# - How to use UFL expressions.

# ## Problem statement

# We will solve the Helmholtz equation subject to a first order absorbing boundary condition:
# $$
# \begin{align*}
# \Delta u + k^2 u &= 0 && \text{in } \Omega,\\
# \nabla u \cdot \mathbf{n} - \mathrm{j}ku &= g && \text{on } \partial\Omega,
# \end{align*}
# $$
# where $k$ is a piecewise constant wavenumber, $\mathrm{j}=\sqrt{-1}$, and $g$ is the boundary source term computed as
# $$g = \nabla u_\text{inc} \cdot \mathbf{n} - \mathrm{j}ku_\text{inc}$$

# +
from utils import plot_mesh, plot_function
from mesh_generation import generate_mesh
from dolfinx.io import gmshio
import IPython
import numpy as np
from mpi4py import MPI

import dolfinx
import ufl
# -

# This example is designed to be executed with complex-valued coefficients.
# To be able to solve this problem, we use the complex build of PETSc.

# +
import sys
from petsc4py import PETSc

if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
    print("This tutorial requires complex number support")
    sys.exit(0)
else:
    print(f"Using {PETSc.ScalarType}.")
# -

# ## Defining model parameters

# +
# wavenumber in free space (air)
k0 = 10 * np.pi

# Corresponding wavelength
lmbda = 2 * np.pi / k0

# Polynomial degree
degree = 6

# Mesh order
mesh_order = 2
# -


# ## Interfacing with GMSH
# We will use Gmsh to generate the computational domain (mesh) for this example.
# As long as Gmsh has been installed (including its Python API), DOLFINx supports direct input of Gmsh models (generated on one process).
# DOLFINx will then in turn distribute the mesh over all processes in the communicator passed to `dolfinx.io.gmshio.model_to_mesh`.

# The function `generate_mesh` creates a Gmsh model and saves it into a .msh file.

# +

# MPI communicator
comm = MPI.COMM_WORLD

file_name = "domain.msh"
generate_mesh(file_name, lmbda, order=mesh_order)
# -

# Now we can read the mesh from file:

# +
mesh, cell_tags, _ = gmshio.read_from_msh(file_name, comm, rank=0, gdim=2)


# -

# ## Material parameters
# In this problem, the wave number in the different parts of the domain
# depends on cell markers, inputted through `cell_tags`. We use the fact that a
# discontinuous Lagrange space of order 0 (cell-wise constants) has a
# one-to-one mapping with the cells local to the process.

# +
W = dolfinx.fem.FunctionSpace(mesh, ("DG", 0))
k = dolfinx.fem.Function(W)
k.x.array[:] = k0
k.x.array[cell_tags.find(1)] = 3 * k0

plot_mesh(mesh, cell_values=k, filename="mesh.html")
IPython.display.HTML(filename="mesh.html")
# -


# ## Boundary source term
# $$g = \nabla u_{inc} \cdot \mathbf{n} - \mathrm{j}ku_{inc}$$
# where $u_{inc} = e^{-jkx}$ the incoming wave, is a plane wave propagating
# in the $x$ direction.
#
# Next, we define the boundary source term by using `ufl.SpatialCoordinate`.
# When using this function, all quantities using this expression will be evaluated
# at quadrature points.

# +
n = ufl.FacetNormal(mesh)
x = ufl.SpatialCoordinate(mesh)
uinc = ufl.exp(1j * k * x[0])
g = ufl.dot(ufl.grad(uinc), n) - 1j * k * uinc
# -

# ## Variational form
# Next, we define the variational problem using a 6th order Lagrange space. 
# Note that as we are using complex valued functions, we have to use the 
# appropriate inner product; see DOLFINx tutorial: Complex numbers for more 
# information. 
#
# Find $u \in V$ such that
# $$-\int_\Omega \nabla u \cdot \nabla \bar{v} ~ dx + \int_\Omega k^2 u \,\bar{v}~ dx - j\int_{\partial \Omega} ku  \bar{v} ~ ds = \int_{\partial \Omega} g \, \bar{v}~ ds \qquad \forall v \in \widehat{V}.$$

# +
element = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), degree)
V = dolfinx.fem.FunctionSpace(mesh, element)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
# -
# +
a = - ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
    + k**2 * ufl.inner(u, v) * ufl.dx \
    - 1j * k * ufl.inner(u, v) * ufl.ds
L = ufl.inner(g, v) * ufl.ds
# - 

# ## Linear solver
# Next, we will solve the problem using a direct solver (LU).

# +
opt = {"ksp_type": "preonly", "pc_type": "lu"}
problem = dolfinx.fem.petsc.LinearProblem(a, L, petsc_options=opt)
uh = problem.solve()
uh.name = "u"
# -

# Visualizing the solution:

# +
plot_function(uh, "uh.html")
IPython.display.HTML(filename="uh.html")
# -

# ### Post-processing with Paraview

# +
from dolfinx.io import XDMFFile, VTXWriter
u_abs = dolfinx.fem.Function(V, dtype=np.float64)
u_abs.x.array[:] = np.abs(uh.x.array)
# -

# Using XDMFFile:

# +
# XDMF writes data to mesh nodes
with XDMFFile(comm, "out.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(u_abs)
# -

# Using VTXWriter

# +
with VTXWriter(comm, "out.bp", [u_abs]) as f:
    f.write(0.0)
# - 

## Homework:

# **Task 1**: download the files `out.xdmf` and `out.bp`.
# Why do they look so different?

# **Task 2**: create a first order Lagrange function and interpolate the solution
# into u1. Use XDMFFile and VTXWriter to visualize the solution.

# +
p1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
V1 = dolfinx.fem.FunctionSpace(mesh, p1)

u1 = dolfinx.fem.Function(V1)
u1.interpolate(uh)
# -

# **Task 3**: Select an iterative solver and plot the solution.
# Can you explain what's happening?