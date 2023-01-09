# # Poisson's problem
# ## Introduction
# 
# In this first tutorial we
# 
# 1.   Present a basic implementation of the finite element solution of Poisson's problem
# 2.   Import all necessary libraries to solve the problem with the FEniCSx platform
# 3.   Solve the problem with different grid refinements and finite element spaces
# 4.   Visualize the solution using pyvista and/or Paraview tools
# 5.   Perform some postprocessing
# 
# 
# **Mathematical formulation:**
# 
# Let us recall the mathematical formulation of the Poisson's problem with
# Dirichlet and Neumann boundary conditions. In differential form: Find ${u} \in \mathcal{C}^2(\Omega)$ such that
# \begin{equation}
# \left \{
# \begin{array}{rcll}
# -\nabla \cdot [ \mu(\mathbf{x}) {u}(\mathbf{x})] & = & f(\mathbf{x}) &  \mbox{in}~\Omega \\
# && \\
# {u}(\mathbf{x}) & = & g_D(\mathbf{x}) &  \mbox{in}~\partial\Omega_D \\
# && \\
# -\mu(\mathbf{x}) \nabla{u}(\mathbf{x})\cdot\check{\mathbf{n}} & = & g_N(\mathbf{x}) &  \mbox{in}~\partial\Omega_N \\
# \end{array}
# \right.
# \end{equation}
# 
# or by multiplying by a sufficiently regular test function $v$ and 
# applying integration by parts. in variational form: Find $u \in V_g(\Omega)$ such that 
# 
# \begin{equation}
# \underbrace{{\int_{\Omega}}{\mu(\mathbf{x})\,\nabla{u}(\mathbf{x})\cdot \nabla{v}(\mathbf{x})}\,dx}_{a(u,v)} =
#         \underbrace{\int_{\Omega}{f(\mathbf{x})\,v(\mathbf{x})}\,dx
#         -\int_{\partial\Omega_N}{g_N(\mathbf{x})\,v(\mathbf{x})}\,ds}_{\ell(v)}~~~\forall v \in V_0(\Omega)
# \end{equation}
# 
# where $a(\cdot,\cdot)$ is a bilinear form and
# $\ell(\cdot)$ is a linear form and the space $V_g$ is
# 
# $$
# V_g = \{v \in H^1(\Omega),~~v(\mathbf{x}) = g_D(\mathbf{x})~\forall \mathbf{x} \in \partial{\Omega}_D  \}
# $$
# 
# Finally, recall that the discrete version of this problem follows from applying the Galerkin method: Find $u_h \in V_{hg} \subset V_g(\Omega)$ such that
# 
# \begin{equation}
# a(u_h,v_h) = \ell(v_h)~~ \forall v_h \in V_{h0} 
# \end{equation}

# ## Initialization
# The first step is to import all necessary libraries. In particular, we must import 
# the [`FEniCSx`](https://fenicsproject.org/) library, which can be done now in Colab thanks to the efforts of the
# [`FEM on Colab`](https://fem-on-colab.github.io/).
# Notice that the first time the library is imported, the system may take a while. Following times are expected to be faster.

#+
try:
  import gmsh
except ImportError:
  !wget "https://github.com/fem-on-colab/fem-on-colab.github.io/raw/7f220b6/releases/gmsh-install.sh" -O "/tmp/gmsh-install.sh" && bash "/tmp/gmsh-install.sh"
  import gmsh

try:
  import dolfinx
except ImportError:
  !wget "https://github.com/fem-on-colab/fem-on-colab.github.io/raw/7f220b6/releases/fenicsx-install-real.sh" -O "/tmp/fenicsx-install.sh" && bash "/tmp/fenicsx-install.sh"
  import dolfinx
#-

# Once the `DOLFINx` package (the main library of the `FEniCSx` project) is installed, we must import some of its modules.
# 
# Relevant `DOLFINx` modules:
# - `dolfinx.mesh`: Classes and functions related to the computational domain
# - `dolfinx.fem`: Finite element method functionality
# - `dolfinx.io`: Input/Output (read/write) functionality
# - `dolfinx.plot`: Convenience functions for exporting plotting data

#+
from dolfinx import mesh, fem, io, plot
#-
