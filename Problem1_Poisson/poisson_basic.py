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

# +
try:
  import matplotlib.pyplot as plt
  import gmsh
except ImportError:
    !wget "https://github.com/fem-on-colab/fem-on-colab.github.io/raw/7f220b6/releases/gmsh-install.sh" - O "/tmp/gmsh-install.sh" & & bash "/tmp/gmsh-install.sh"
    import gmsh

try:
    import dolfinx
except ImportError:
    !wget "https://github.com/fem-on-colab/fem-on-colab.github.io/raw/7f220b6/releases/fenicsx-install-real.sh" - O "/tmp/fenicsx-install.sh" & & bash "/tmp/fenicsx-install.sh"
    import dolfinx
# -

# Once the `DOLFINx` package (the main library of the `FEniCSx` project) is installed, we must import some of its modules.
#
# Relevant `DOLFINx` modules:
# - `dolfinx.mesh`: Classes and functions related to the computational domain
# - `dolfinx.fem`: Finite element method functionality
# - `dolfinx.io`: Input/Output (read/write) functionality
# - `dolfinx.plot`: Convenience functions for exporting plotting data

# +
from dolfinx import mesh, fem, io, plot
# -

# We must also import the [`Unified Form Language`](https://github.com/FEniCS/ufl/) (`UFL`) library
# which provides us a specific language for writting variational formulations as well as the required
# mathematical operators

# +
from ufl import SpatialCoordinate, TestFunction, TrialFunction, Measure, dx, ds, grad, inner, as_vector, FacetNormal
# -

# Finally, we import some auxiliary libraries:

# +
import numpy as np
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
# -

# ## Problem definition
#
# The first thing is to define the computational domain and parts of the boundary on which the Dirichlet conditions are to be applied:

# +
# Computational mesh
Lx, Ly = 1.0, 1.0
msh = mesh.create_rectangle(comm=MPI.COMM_WORLD,
                            points=((0.0, 0.0), (Lx, Ly)), n=(64, 64),
                            cell_type=mesh.CellType.triangle)
# -

# ## Finite element solution

# We must create now the discrete function space associated to a conforming finite element partition $\mathcal{T}_h$ of $\Omega$,
# ***the mesh***, in which the solution is sought.
#
# For this problem, the natural choice is a space of continuous functions, that are elementwise polynomials of degree $k$
#
# $$
# V(\mathcal{T}_h) = V_h = \{v \in H ^ 1(\Omega), ~v | _E \in P_k(E) \, \forall E \in mathcal{T}_h\}
# $$

# +
degree = 1
V = fem.FunctionSpace(msh, ("CG", degree))
# -

# Above, `CG` refers to Continuous Galerkin, which is just another name for the widely used * Lagrange * spaces in
# the FEM, so, the following would be equivalent:
#
#    V = fem.FunctionSpace(msh, ("Lagrange", degree))
#
# Now, we identify the parts of the boundary on which
# Dirichlet conditions are given and define the corresponding objects.
# In `FEniCSx` we can identify degress of freedom located on the boundary:
# *   geometrically, or
# *   topologically
#
# in this case, we use geometric detection:

# +
# Boundary values
uleft = 100.0
uright = 1.0

# Identify the degrees of freedom to which the values are to be applied
dofsL = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0))
dofsR = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], Lx))

# Define the Dirichlet type BCs as a list
bcs = [fem.dirichletbc(ScalarType(uleft), dofsL, V),
                       fem.dirichletbc(ScalarType(uright), dofsR, V)]
# -

# Finally, we can write the variational formulation of our problem and solve, recalling
# $$
# a(u, v) = \int_{\Omega}{\mu\, \nabla{u}\cdot \nabla{v}}\, dx
# $$
# and , if we take $g_N = 0$, we naturally have
# $$
# \ell(v) = \int_{\Omega}{f\, v}\, dx
# $$

# +
# Variational formulation
# Trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Bilinear form
mu = fem.Constant(msh, 1.0)
a = inner(mu*grad(u), grad(v)) * dx

# Linear form
f = 1.0
source = fem.Constant(msh, f)
L = source * v * dx

# Solve the problem
problem = fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options={
                                  "ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()
# -

# ## Visualization and postprocessing

# +
uh.name = "Temperature"
with io.XDMFFile(MPI.COMM_WORLD, "temperature.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh)
    xdmf.write_function(uh)
# -

# A typical verification is to compute the relevant error norms, which in this case are the $L^2(\Omega)$ error norm:
#
# $$
# e_{L^2(\Omega)} = \sqrt{ \int_{\Omega}{(u - u_h)^2}\,dx}
# $$
# and the $H^1(\Omega)$ error norm
# $$
# e_{H^1(\Omega)} = \sqrt{ \int_{\Omega}{\left [ (u - u_h)^2 + \nabla{(u-u_h)} \cdot \nabla{(u-u_h)}\right ] }\,dx }
# $$
#
# As a result of taking zero Neumann boundary conditions on the top
# and bottom walls, the exact solution to this problem, with $\mu$ and $f$ constants, ends up being dependent only on the $x$ coordinate, i.e., we are actually solving the problem
#
# $$
# -\dfrac{d^2u}{dx^2} = f,~~u(0) = u_l,~~u(L_x) = u_r
# $$
#
# whose solution is
#
# $$
# u = u_l + \left ( \dfrac{u_r - u_l}{L_x} + \frac12 \, f \, L_x \right ) x - \frac12 f  x^2
# $$

# +
# We need the coordinates to create a ufl expression
x = SpatialCoordinate(msh)

# to define the exact solution


def uex(x): return uleft + ((uright - uleft) /
                            Lx + 0.5*f*Lx)*x[0] - 0.5*f*x[0]**2


# and the error
eh = uh - uex(x)

# Now compute the L2 and H1 error norms
eL2form = fem.form(eh**2*dx)
eH1form = fem.form(inner(grad(eh), grad(eh))*dx)

errorL2 = fem.assemble_scalar(eL2form)
errorH1 = errorL2 + fem.assemble_scalar(eH1form)

print("    |-L2error=", np.sqrt(errorL2))
print("    |-H1error=", np.sqrt(errorH1))
# -

# ## A sanity check: Convergence rates
#
# How do we know the numerical solution is actually correct?

# We can compute the rate of convergence of the numerical method and compare to the expected asymptotic value.
# To that end, we first encapsulate the whole resolution into a single python function that
# takes as arguments the desired refinement and degree of the
# polynomial space and returns the $L^2(\Omega)$ and $H^1(\Omega)$-error norms, i.e.

# +
def SolvePoisson(msh):
    degree = 1
    V = fem.FunctionSpace(msh, ("CG", degree))

    nunks = V.dofmap.index_map.size_global
    print("Solving: Characteristic mesh size h=%f (%d elements, %d unknowns)" %
          (Lx/N, 2*N*N, nunks))

    dofsL = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0))
    dofsR = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], Lx))
    bcs = [fem.dirichletbc(ScalarType(uleft), dofsL, V),
           fem.dirichletbc(ScalarType(uright), dofsR, V)]

    u, v = TrialFunction(V), TestFunction(V)
    mu = fem.Constant(msh, 1.0)
    a = inner(mu*grad(u), grad(v)) * dx
    f = 1.0
    source = fem.Constant(msh, f)
    L = source * v * dx
    problem = fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options={
                                      "ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    def uex(x): return uleft + ((uright - uleft) /
                                Lx + 0.5*f*Lx)*x[0] - 0.5*f*x[0]**2
    x = SpatialCoordinate(msh)
    eh = uh - uex(x)
    eL2form = fem.form(eh**2*dx)
    eH1form = fem.form(inner(grad(eh), grad(eh))*dx)
    errorL2 = fem.assemble_scalar(eL2form)
    errorH1 = errorL2 + fem.assemble_scalar(eH1form)

    return np.sqrt(errorL2), np.sqrt(errorH1)

# Make a loop over the meshes
Lx = Ly = 1.0
degree = 1
eL2, eH1, h = [], [], []
for N in [10, 20, 40, 80, 160, 320]:
    msh = mesh.create_rectangle(comm=MPI.COMM_WORLD,
                                points=((0.0, 0.0), (Lx, Ly)), n=(N, N),
                                cell_type=mesh.CellType.triangle)

    errorL2, errorH1 = SolvePoisson(msh)
    eL2.append(errorL2)
    eH1.append(errorH1)
    h.append(Lx/N)

plt.figure(figsize=(9, 6))
plt.rcParams['font.size'] = '16'
plt.axes(facecolor="lightgray")
plt.xlabel('$h$', fontsize=20)
plt.ylabel('Error', fontsize=20)
l = plt.plot(h, eL2, '-or', h, eH1, '-ob')
plt.setp(l, 'markersize', 10)
plt.setp(l, 'markerfacecolor', 'white')
plt.legend(['$e_{L^2}$', '$e_{H1}$'])
plt.loglog(base=10)
plt.grid()
plt.show()
# -