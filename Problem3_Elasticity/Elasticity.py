# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + [markdown] id="HlwXpyx788Vo"
# # Solid mechanics: Linear elasticity
# ---

# + [markdown] id="sUiY4ZupvRWL"
# ## Introduction

# + [markdown] id="04Q-hHB9299K"
# In this tutorial we
#
# 1.   Present an implementation of the finite element discretization
# of the Navier-Poisson elastostatic problem
# 2.   Create a non-trivial geometry and impose essential boundary conditions
# on a vector field
# 3.   Visualize the solution using Paraview
# 4.   Perform some postprocessing of the solution.
#
# **Mathematical formulation:**
#
# \begin{equation}
# \left \{
# \begin{array}{rcll}
# -\nabla \cdot \boldsymbol{\sigma} (\mathbf{u}) & = & \mathbf{f} & \mbox{in}~\Omega \\
# & & & \\
# \mathbf{u} & = & \mathbf{g} & \mbox{on}~\Gamma_{\mathbf{u}} \\
# & & & \\
# \boldsymbol{\sigma} \cdot \check{\mathbf{n}} & = & \boldsymbol{\mathcal{F}} & \mbox{on}~\Gamma_{\boldsymbol{\mathcal{F}}}
# \end{array}
# \right.
# \end{equation}
# where the stress tensor is
#
# $$
# \boldsymbol{\sigma} = 2\mu\, \boldsymbol{\varepsilon}({\mathbf{u}})+ \lambda \left ( \nabla\cdot\mathbf{u} \right )\, \mathbf{I}_{d \times d}
# $$
#  
# where $d$ is the spatial dimension, $\mathbf{I}_{d \times d}$ is 
# the identity tensor and the deformation tensor is defined by
#
# $$
# \boldsymbol{\varepsilon}({\mathbf{u}}) = \frac12 (\nabla{\mathbf{u}} + \nabla^{\intercal}{\mathbf{u}})
# $$
#
# Introducing the space of kinematically admissible motions
#
# $$
# V_{\mathbf{g}} = \{\mathbf{v} \in \left [ H^1(\Omega) \right ]^d,~\mathbf{v} = \mathbf{g}~\mbox{on}~\Gamma_D\}
# $$
#
# and applying the principle of virtual work, the variational formulation is obtained: Find $\mathbf{u} \in V_{\mathbf{g}}$ such that
#
# \begin{eqnarray}
# \underbrace{\int_{\Omega}{\left [2\,\mu \boldsymbol{\varepsilon}(\mathbf{u}) : \boldsymbol{\varepsilon}(\mathbf{v})
# # + \lambda\, (\nabla \cdot \mathbf{u})\,(\nabla \cdot \mathbf{v}) \right ]\,dx}}_{a(\mathbf{u},\mathbf{v})} =
#         \underbrace{\int_{\Omega}{\mathbf{f}\cdot \mathbf{v}}\,dx +
# \int_{\Gamma_{\boldsymbol{\mathcal{F}}}}{\boldsymbol{\mathcal{F}} \cdot \mathbf{v}}\,ds}_{\ell(\mathbf{v})}
# \end{eqnarray}
# $\forall \mathbf{v} \in V_{\mathbf{0}}$.
#
# Finally, recall that the discrete version of this problem follows from applying the Galerkin method: Find $\mathbf{u}_h \in V_{h\mathbf{g}} \subset V_{\mathbf{g}}(\Omega)$ such that
#
# \begin{equation}
# a(\mathbf{u}_h,\mathbf{v}_h) = \ell(\mathbf{v}_h)~~ \forall \mathbf{v}_h \in V_{h\mathbf{0}} 
# \end{equation}

# + [markdown] id="7TeY3nHjqzod"
# ## Initialization

# + [markdown] id="KV_eRNQpHfOv"
# As in previous tutorials, we import all necessary libraries, namely, `gmsh`, `dolfinx` and `ufl`

# + id="69Xzz1wQx-Nd"
try:
  import gmsh
except ImportError:
  # !wget "https://github.com/fem-on-colab/fem-on-colab.github.io/raw/7f220b6/releases/gmsh-install.sh" -O "/tmp/gmsh-install.sh" && bash "/tmp/gmsh-install.sh"
  import gmsh

try:
  import dolfinx
except ImportError:
  # !wget "https://github.com/fem-on-colab/fem-on-colab.github.io/raw/7f220b6/releases/fenicsx-install-real.sh" -O "/tmp/fenicsx-install.sh" && bash "/tmp/fenicsx-install.sh"
  import dolfinx

try:
    import pyvista
except ImportError:
    # !pip install - q piglet pyvirtualdisplay ipyvtklink pyvista panel
    # !apt-get - qq install xvfb
    import pyvista

# + id="ExTIMkkrxi-H"
from dolfinx import mesh, fem, io, plot
from ufl import SpatialCoordinate, TestFunction, TrialFunction, Measure, Identity, div, dx, ds, grad, nabla_grad, inner, sym, tr, sqrt, as_vector, FacetNormal

import numpy as np
from mpi4py import MPI
from petsc4py.PETSc import ScalarType


# + [markdown] id="DmgHdjHcaPJF"
# Now, we create a mesh using 

# + id="fDoVkR60ydkP" colab={"resources": {"http://localhost:8080/jupyter-threejs.js": {"data": "CjwhRE9DVFlQRSBodG1sPgo8aHRtbCBsYW5nPWVuPgogIDxtZXRhIGNoYXJzZXQ9dXRmLTg+CiAgPG1ldGEgbmFtZT12aWV3cG9ydCBjb250ZW50PSJpbml0aWFsLXNjYWxlPTEsIG1pbmltdW0tc2NhbGU9MSwgd2lkdGg9ZGV2aWNlLXdpZHRoIj4KICA8dGl0bGU+RXJyb3IgNDA0IChOb3QgRm91bmQpISExPC90aXRsZT4KICA8c3R5bGU+CiAgICAqe21hcmdpbjowO3BhZGRpbmc6MH1odG1sLGNvZGV7Zm9udDoxNXB4LzIycHggYXJpYWwsc2Fucy1zZXJpZn1odG1se2JhY2tncm91bmQ6I2ZmZjtjb2xvcjojMjIyO3BhZGRpbmc6MTVweH1ib2R5e21hcmdpbjo3JSBhdXRvIDA7bWF4LXdpZHRoOjM5MHB4O21pbi1oZWlnaHQ6MTgwcHg7cGFkZGluZzozMHB4IDAgMTVweH0qID4gYm9keXtiYWNrZ3JvdW5kOnVybCgvL3d3dy5nb29nbGUuY29tL2ltYWdlcy9lcnJvcnMvcm9ib3QucG5nKSAxMDAlIDVweCBuby1yZXBlYXQ7cGFkZGluZy1yaWdodDoyMDVweH1we21hcmdpbjoxMXB4IDAgMjJweDtvdmVyZmxvdzpoaWRkZW59aW5ze2NvbG9yOiM3Nzc7dGV4dC1kZWNvcmF0aW9uOm5vbmV9YSBpbWd7Ym9yZGVyOjB9QG1lZGlhIHNjcmVlbiBhbmQgKG1heC13aWR0aDo3NzJweCl7Ym9keXtiYWNrZ3JvdW5kOm5vbmU7bWFyZ2luLXRvcDowO21heC13aWR0aDpub25lO3BhZGRpbmctcmlnaHQ6MH19I2xvZ297YmFja2dyb3VuZDp1cmwoLy93d3cuZ29vZ2xlLmNvbS9pbWFnZXMvbG9nb3MvZXJyb3JwYWdlL2Vycm9yX2xvZ28tMTUweDU0LnBuZykgbm8tcmVwZWF0O21hcmdpbi1sZWZ0Oi01cHh9QG1lZGlhIG9ubHkgc2NyZWVuIGFuZCAobWluLXJlc29sdXRpb246MTkyZHBpKXsjbG9nb3tiYWNrZ3JvdW5kOnVybCgvL3d3dy5nb29nbGUuY29tL2ltYWdlcy9sb2dvcy9lcnJvcnBhZ2UvZXJyb3JfbG9nby0xNTB4NTQtMngucG5nKSBuby1yZXBlYXQgMCUgMCUvMTAwJSAxMDAlOy1tb3otYm9yZGVyLWltYWdlOnVybCgvL3d3dy5nb29nbGUuY29tL2ltYWdlcy9sb2dvcy9lcnJvcnBhZ2UvZXJyb3JfbG9nby0xNTB4NTQtMngucG5nKSAwfX1AbWVkaWEgb25seSBzY3JlZW4gYW5kICgtd2Via2l0LW1pbi1kZXZpY2UtcGl4ZWwtcmF0aW86Mil7I2xvZ297YmFja2dyb3VuZDp1cmwoLy93d3cuZ29vZ2xlLmNvbS9pbWFnZXMvbG9nb3MvZXJyb3JwYWdlL2Vycm9yX2xvZ28tMTUweDU0LTJ4LnBuZykgbm8tcmVwZWF0Oy13ZWJraXQtYmFja2dyb3VuZC1zaXplOjEwMCUgMTAwJX19I2xvZ297ZGlzcGxheTppbmxpbmUtYmxvY2s7aGVpZ2h0OjU0cHg7d2lkdGg6MTUwcHh9CiAgPC9zdHlsZT4KICA8YSBocmVmPS8vd3d3Lmdvb2dsZS5jb20vPjxzcGFuIGlkPWxvZ28gYXJpYS1sYWJlbD1Hb29nbGU+PC9zcGFuPjwvYT4KICA8cD48Yj40MDQuPC9iPiA8aW5zPlRoYXTigJlzIGFuIGVycm9yLjwvaW5zPgogIDxwPiAgPGlucz5UaGF04oCZcyBhbGwgd2Uga25vdy48L2lucz4K", "ok": false, "headers": [["content-length", "1449"], ["content-type", "text/html; charset=utf-8"]], "status": 404, "status_text": ""}}, "base_uri": "https://localhost:8080/", "height": 521} executionInfo={"status": "ok", "timestamp": 1674214789414, "user_tz": 0, "elapsed": 4195, "user": {"displayName": "Igor Baratta", "userId": "07060436923309155079"}} outputId="a661d772-2e29-4b70-f83d-469a5e6844f7"
def GenerateMesh():
  
  gmsh.initialize()
  proc = MPI.COMM_WORLD.rank
  if proc == 0:
      lc = 0.05
      Db   = 0.4
      Hb   = 0.4
      Hp   = 6*Hb
      R    = 3*Hb
      TT   = np.sqrt(R*R - 4*Hb*Hb)
      
      gmsh.model.geo.addPoint(0, 0, 0, lc, 1)
      gmsh.model.geo.addPoint(Db, 0, 0, lc, 2)
      gmsh.model.geo.addPoint(Db, Hb, 0, 0.5*lc, 3)
      gmsh.model.geo.addPoint(TT+Db, 3*Hb, 0, lc, 4)
      gmsh.model.geo.addPoint(Db, 5*Hb, 0, lc, 5)
      gmsh.model.geo.addPoint(Db, 6*Hb, 0, 0.5*lc, 6)
      gmsh.model.geo.addPoint(0, 6*Hb, 0, lc, 7)
      gmsh.model.geo.addPoint(0, 3*Hb, 0, 0.1*lc, 8)
      gmsh.model.geo.addPoint(TT+Db-R, 3*Hb, 0, 0.1*lc, 9)
      
      gmsh.model.geo.addLine(1, 2, 1)
      gmsh.model.geo.addLine(2, 3, 2)

      gmsh.model.geo.addCircleArc(3, 4, 9, 3)
      gmsh.model.geo.addCircleArc(9, 4, 5, 4)
      
      gmsh.model.geo.addLine(5, 6, 5)
      gmsh.model.geo.addLine(6, 7, 6)
      gmsh.model.geo.addLine(7, 8, 7)
      gmsh.model.geo.addLine(8, 1, 8)
      
      gmsh.model.geo.addCurveLoop([1, 2, 3, 4, 5, 6, 7, 8], 1)
      gmsh.model.geo.addPlaneSurface([1], 1)
      gmsh.model.geo.synchronize()
      # Tag the whole boundary with 101
      gmsh.model.addPhysicalGroup(1, [1, 2, 3, 4, 5, 6, 7, 8], 101)
      # Tag the top boundary with 100
      gmsh.model.addPhysicalGroup(1, [6], 100)
      ps = gmsh.model.addPhysicalGroup(2, [1])
      gmsh.model.setPhysicalName(2, ps, "My surface") 
      gmsh.model.geo.synchronize()
  
  gmsh.option.setNumber("Mesh.Algorithm", 6)
  gmsh.model.mesh.generate(2)
  msh, subdomains, boundaries = io.gmshio.model_to_mesh(gmsh.model, comm=MPI.COMM_WORLD, rank=0, gdim=2)
  gmsh.finalize()
  return msh, subdomains, boundaries

msh, subdomains, boundaries = GenerateMesh()

with io.XDMFFile(MPI.COMM_WORLD, "body.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh)

import IPython

def plot_mesh(mesh, filename="file.html"):
    pyvista.start_xvfb()
    grid = pyvista.UnstructuredGrid(*plot.create_vtk_mesh(mesh))
    plotter = pyvista.Plotter(notebook=True, window_size=[500,500])
    plotter.add_mesh(grid, show_edges=True)
    plotter.camera.zoom(4.0)
    plotter.view_xy()
    plotter.export_html(filename, backend="pythreejs")
    plotter.close()


plot_mesh(msh, "mesh.html")
IPython.display.HTML(filename="mesh.html")

# + [markdown] id="GHEnrW5_dXPQ"
# ## Finite element solution

# + [markdown] id="_w-zEZ7fdloa"
# We must create now the discrete function space associated to the mesh $\mathcal{T}_h$. As in previous examples a natural choice is a space of continuous vector functions, whose components are elementwise polynomials of degree $k$
#
# $$
# V(\mathcal{T}_h) = V_h = \{\mathbf{v} \in [H^1(\Omega)]^d,~\mathbf{v}|_E \in [P_k(E)]^d \, \forall E \in \mathcal{T}_h\}
# $$
#
# which is done in `dolfinx` using

# + id="35oDBR1Oeusx"
degree = 1
V = fem.VectorFunctionSpace(msh, ("CG", 1))

# + [markdown] id="paubzY5efWjl"
# As usual, setting the boundary conditions is the step that takes more work. We must identify the degrees of freedom on the boundary and set accordingly. 
# For the problem at hand we will consider the following conditions
#
# \begin{eqnarray}
# \mathbf{u} & = & (0,0)^{\intercal} ~~\mbox{in}~~\Gamma_{\mbox{bottom}} \\
# & & \\
# u_x & = & 0~~\mbox{in}~~\Gamma_{\mbox{left}}
# \end{eqnarray}
#
# These conditions ensure that **rigid body** motions (rotations and translations) are totally restricted.

# + id="fKejVTE43cnd"
u_bottom = ScalarType((0.0, 0.0))
ux_left   = ScalarType(0.0)

# For the left boundary, just restrict u_x
sdim = msh.topology.dim
fdim = sdim - 1
facets_left = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
dofsL = fem.locate_dofs_topological(V.sub(0), fdim, facets_left)

# For the bottom restrict everything
dofsB = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[1], 0.0))
bcs = [fem.dirichletbc(u_bottom, dofsB, V), fem.dirichletbc(ux_left, dofsL, V.sub(0))]

# + [markdown] id="UJgkqWKgf0A2"
# As for the natural bounday conditions, on the top wall we will apply 
# the following surface force distribution
#
# $$
# \boldsymbol{\mathcal{F}} = (0,-0.1)^{\intercal}
# $$
# so as to impose a compressive load. The rest of the boundary is traction free and the body forces are considered to be negligible, 
#
# $$
# \boldsymbol{\mathcal{F}} = (0,0)^{\intercal},~~~\mathbf{f} = (0,0)^{\intercal}
# $$
#
# so, we can finally define the bilinear and linear forms and write the variational formulation of the elastostatic problem
#
# $$
# a(\mathbf{u},\mathbf{v}) = \int_{\Omega}{\left [2\mu \,\boldsymbol{\varepsilon}(\mathbf{u}) : \boldsymbol{\varepsilon}(\mathbf{v})
# # + \lambda\, (\nabla \cdot \mathbf{u})\,(\nabla \cdot \mathbf{v}) \right ]\,dx}
# $$
#
# and
#
# $$
# \ell(\mathbf{v}) = \int_{\Omega}{\mathbf{f}\cdot \mathbf{v}}\,dx +
# \int_{\Gamma_{\boldsymbol{\mathcal{F}}}}{\boldsymbol{\mathcal{F}} \cdot \mathbf{v}}\,ds
# $$

# + colab={"base_uri": "https://localhost:8080/"} id="Q3F5qNVkfynZ" executionInfo={"status": "ok", "timestamp": 1674217195349, "user_tz": 0, "elapsed": 1198, "user": {"displayName": "Igor Baratta", "userId": "07060436923309155079"}} outputId="84451108-b66e-4437-a38e-b7d1af7e5712"
# The rest of the boundary is traction free, except for the top in which we apply a surface force distribution

# surface force
F = fem.Constant(msh, ScalarType( (0.0, -0.1) ) )

# Body force
f = fem.Constant(msh, ScalarType( (0.0, 0.0) ) )

# Constitutive parameters
E, nu = 10.0, 0.3
mu    = E/(2.0*(1.0 + nu))
lamb  = E*nu/((1.0 + nu)*(1.0 - 2.0*nu))

u, v = TrialFunction(V), TestFunction(V)

def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)

def sigma(u):
    return lamb*div(u)*Identity(sdim) + 2*mu*epsilon(u)

x = SpatialCoordinate(msh)

ds = Measure("ds")(subdomain_data=boundaries)

a = inner(sigma(u), epsilon(v)) * dx
L = inner(f, v)*dx + inner(F,v)*ds(100)

petsc_opts={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps", "ksp_monitor": None}
problem = fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options=petsc_opts)
uh = problem.solve()

# + [markdown] id="wT8b_QXub5pO"
# ## Visualization and postprocessing

# + [markdown] id="pZ0T61UH6TNW"
# Let us write the solution for visualization in `Paraview` as we have done in the previous examples

# + id="saVLTaLwfKoO"
uh.name = "displacement"
with io.XDMFFile(MPI.COMM_WORLD, "displacement.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh)
    xdmf.write_function(uh)

from google.colab import files
files.download('displacement.xdmf') 
files.download('displacement.h5')


# + [markdown] id="4FiWe7UsbwGD"
# ## Homework 3

# + [markdown] id="mP251voMb2Tn"
# 1. **Von Mises stresses**
#
# Given the deviatoric stresses 
# $$
# \boldsymbol{s} = \boldsymbol{\sigma}(\mathbf{u}_h) - \frac{\mbox{tr}(\boldsymbol{\sigma}(\mathbf{u}_h))}{d}\boldsymbol{I}_{d\times d}
# $$
#
# Compute the **scalar** quantity known as the Von Mises stresses
# defined as the second invariant of the deviatoric stresses:
#
# $$
# \sigma_V = \sqrt{\frac32\boldsymbol{s}:\boldsymbol{s}}
# $$
#
# where $:$ stands dor the double contraction or scalar product between matrizes.
# This quantity is used by engineers to detect the critical parts of the structure.
#
# Implement in `dolfinx`. For visualization of results, interpolate $\sigma_V$ onto a space of elementwise constant functions (a `DG` space of order 0) as we have introduced before
#
# $$
# Q_h = \{v \in L^2(\Omega),~v|_E \in P_0(E) \, \forall E \in \mathcal{T}_h\}
# $$
#
# Follow the next guidelines:
#
#     s = sigma(uh) - ...
#     sigmaV = sqrt(...)
#
#     Q = fem.FunctionSpace(msh, ("DG", 0))
#     vM_expr = fem.Expression(sigmaV, Q.element.interpolation_points())
#     vonMises = fem.Function(Q)
#     vonMises.interpolate(vM_expr)
#
#     stresses.name = "von_Mises"
#     with io.XDMFFile(msh.comm, "vonmises.xdmf", "w") as file:
#       file.write_mesh(msh)
#       file.write_function(vonMises)
#
#     from google.colab import files
#     files.download('vonmises.xdmf') 
#     files.download('vonmises.h5')
#
# Notice the use of the `interpolate` method to assign to each element the corresponding value of $\sigma_V$.

# + [markdown] id="GU-xKNZxBTlS"
# 2. **OPTIONAL: 3D problem**
#
# Consider the 3D version of the previous problem which is shown
# in the figure below. This mesh can be created with the function `GenerateMesh3D()`. 
#
# Implement the necessary changes to solve
# the problem with the following boundary conditions
#
# \begin{eqnarray}
# \mathbf{u} & = & (0,0,0)^{\intercal} ~~\mbox{in}~~\Gamma_{\mbox{bottom}} \nonumber \\
# & & \nonumber \\
# \mathbf{u} & = & (0,0,-0.1)^{\intercal} ~~\mbox{in}~~\Gamma_{\mbox{top}} \nonumber
# \end{eqnarray}
#
# whereas the rest of the boundary remains traction free 
#
# $$
# \boldsymbol{\mathcal{F}} = (0, 0, 0)^{\intercal}
# $$

# + id="Q3k93hsUilE7"
def GenerateMesh3D():
  gmsh.initialize()
  proc = MPI.COMM_WORLD.rank
  if proc == 0:

      lc = 0.025
      Db = 0.4
      Hb =  0.4
      global Hp
      Hp = 6*Hb
      R  = 3*Hb
      TT = np.sqrt(R*R - 4*Hb*Hb)
      
      gmsh.model.geo.addPoint(0, 0, 0, lc, 1)
      gmsh.model.geo.addPoint(Db, 0, 0, lc, 2)
      gmsh.model.geo.addPoint(Db, Hb, 0, 0.5*lc, 3)
      gmsh.model.geo.addPoint(TT+Db, 3*Hb, 0, lc, 4)
      gmsh.model.geo.addPoint(Db, 5*Hb, 0, lc, 5)
      gmsh.model.geo.addPoint(Db, 6*Hb, 0, 0.5*lc, 6)
      gmsh.model.geo.addPoint(0, 6*Hb, 0, lc, 7)
      gmsh.model.geo.addPoint(0, 3*Hb, 0, 0.1*lc, 8)
      gmsh.model.geo.addPoint(TT+Db-R, 3*Hb, 0, 0.1*lc, 9)
      
      gmsh.model.geo.addLine(1, 2, 1)
      gmsh.model.geo.addLine(2, 3, 2)
      gmsh.model.geo.addCircleArc(3, 4, 9, 3)
      gmsh.model.geo.addCircleArc(9, 4, 5, 4)
      gmsh.model.geo.addLine(5, 6, 5)
      gmsh.model.geo.addLine(6, 7, 6)
      gmsh.model.geo.addLine(7, 8, 7)
      gmsh.model.geo.addLine(8, 1, 8)
      
      gmsh.model.geo.addCurveLoop([1, 2, 3, 4, 5, 6, 7, 8], 1)
      gmsh.model.geo.addPlaneSurface([1], 1)
      gmsh.model.geo.synchronize()
      gmsh.model.addPhysicalGroup(1, [1, 2, 3, 4, 5, 6, 7, 8], 101)
      ps = gmsh.model.addPhysicalGroup(2, [1])
      gmsh.model.setPhysicalName(2, ps, "My surface 1")

      gmsh.model.geo.addPoint(-Db, 0, 0, lc, 10)
      gmsh.model.geo.addPoint(-Db, Hb, 0, 0.5*lc, 11)
      gmsh.model.geo.addPoint(-(TT+Db), 3*Hb, 0, lc, 12)
      gmsh.model.geo.addPoint(-Db, 5*Hb, 0, lc, 13)
      gmsh.model.geo.addPoint(-Db, 6*Hb, 0, 0.5*lc, 14)
      gmsh.model.geo.addPoint(-(TT+Db-R), 3*Hb, 0, 0.1*lc, 15)
      
      gmsh.model.geo.addLine(1, 8, 9)
      gmsh.model.geo.addLine(8, 7, 10)
      gmsh.model.geo.addLine(7, 14, 11)
      gmsh.model.geo.addLine(14, 13, 12)
      gmsh.model.geo.addCircleArc(13, 12, 15, 13)
      gmsh.model.geo.addCircleArc(15, 12, 11, 14)
      gmsh.model.geo.addLine(11, 10, 15)
      gmsh.model.geo.addLine(10, 1, 16)
      
      gmsh.model.geo.addCurveLoop([9, 10, 11, 12, 13, 14, 15, 16], 2)
      gmsh.model.geo.addPlaneSurface([2], 2)
      gmsh.model.geo.synchronize()
      gmsh.model.addPhysicalGroup(1, [9, 10, 11, 12, 13, 14, 15, 16], 103)
      ps = gmsh.model.addPhysicalGroup(2, [2])

      gmsh.model.setPhysicalName(2, ps, "My surface 2")
      gmsh.model.geo.synchronize()

      ov1 = gmsh.model.geo.revolve([(2, 1)], 0, 0, 0, 0, 1, 0, -np.pi / 2)
      ov2 = gmsh.model.geo.revolve([(2, 1)], 0, 0, 0, 0, 1, 0,  np.pi / 2)
      ov3 = gmsh.model.geo.revolve([(2, 2)], 0, 0, 0, 0, 1, 0, -np.pi / 2)
      ov4 = gmsh.model.geo.revolve([(2, 2)], 0, 0, 0, 0, 1, 0,  np.pi / 2)
      gmsh.model.geo.synchronize()

      gmsh.model.addPhysicalGroup(3, [ov1[1][1]], 105)
      gmsh.model.addPhysicalGroup(3, [ov2[1][1]], 106)
      gmsh.model.addPhysicalGroup(3, [ov3[1][1]], 107)
      gmsh.model.addPhysicalGroup(3, [ov4[1][1]], 108)
      gmsh.model.geo.synchronize()
      
  gmsh.option.setNumber("Mesh.Algorithm", 2)
  gmsh.model.mesh.generate(3)
  #gmsh.write("./3dcorpo.msh")
  #gmsh.write("foo.geo_unrolled")
  msh, subdomains, boundaries = io.gmshio.model_to_mesh(gmsh.model, comm=MPI.COMM_WORLD, rank=0, gdim=3)
  gmsh.finalize()
  return msh, subdomains, boundaries

msh, subdomains, boundaries = GenerateMesh3D()

with io.XDMFFile(MPI.COMM_WORLD, "3Dbody.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh)


# + colab={"resources": {"http://localhost:8080/jupyter-threejs.js": {"data": "CjwhRE9DVFlQRSBodG1sPgo8aHRtbCBsYW5nPWVuPgogIDxtZXRhIGNoYXJzZXQ9dXRmLTg+CiAgPG1ldGEgbmFtZT12aWV3cG9ydCBjb250ZW50PSJpbml0aWFsLXNjYWxlPTEsIG1pbmltdW0tc2NhbGU9MSwgd2lkdGg9ZGV2aWNlLXdpZHRoIj4KICA8dGl0bGU+RXJyb3IgNDA0IChOb3QgRm91bmQpISExPC90aXRsZT4KICA8c3R5bGU+CiAgICAqe21hcmdpbjowO3BhZGRpbmc6MH1odG1sLGNvZGV7Zm9udDoxNXB4LzIycHggYXJpYWwsc2Fucy1zZXJpZn1odG1se2JhY2tncm91bmQ6I2ZmZjtjb2xvcjojMjIyO3BhZGRpbmc6MTVweH1ib2R5e21hcmdpbjo3JSBhdXRvIDA7bWF4LXdpZHRoOjM5MHB4O21pbi1oZWlnaHQ6MTgwcHg7cGFkZGluZzozMHB4IDAgMTVweH0qID4gYm9keXtiYWNrZ3JvdW5kOnVybCgvL3d3dy5nb29nbGUuY29tL2ltYWdlcy9lcnJvcnMvcm9ib3QucG5nKSAxMDAlIDVweCBuby1yZXBlYXQ7cGFkZGluZy1yaWdodDoyMDVweH1we21hcmdpbjoxMXB4IDAgMjJweDtvdmVyZmxvdzpoaWRkZW59aW5ze2NvbG9yOiM3Nzc7dGV4dC1kZWNvcmF0aW9uOm5vbmV9YSBpbWd7Ym9yZGVyOjB9QG1lZGlhIHNjcmVlbiBhbmQgKG1heC13aWR0aDo3NzJweCl7Ym9keXtiYWNrZ3JvdW5kOm5vbmU7bWFyZ2luLXRvcDowO21heC13aWR0aDpub25lO3BhZGRpbmctcmlnaHQ6MH19I2xvZ297YmFja2dyb3VuZDp1cmwoLy93d3cuZ29vZ2xlLmNvbS9pbWFnZXMvbG9nb3MvZXJyb3JwYWdlL2Vycm9yX2xvZ28tMTUweDU0LnBuZykgbm8tcmVwZWF0O21hcmdpbi1sZWZ0Oi01cHh9QG1lZGlhIG9ubHkgc2NyZWVuIGFuZCAobWluLXJlc29sdXRpb246MTkyZHBpKXsjbG9nb3tiYWNrZ3JvdW5kOnVybCgvL3d3dy5nb29nbGUuY29tL2ltYWdlcy9sb2dvcy9lcnJvcnBhZ2UvZXJyb3JfbG9nby0xNTB4NTQtMngucG5nKSBuby1yZXBlYXQgMCUgMCUvMTAwJSAxMDAlOy1tb3otYm9yZGVyLWltYWdlOnVybCgvL3d3dy5nb29nbGUuY29tL2ltYWdlcy9sb2dvcy9lcnJvcnBhZ2UvZXJyb3JfbG9nby0xNTB4NTQtMngucG5nKSAwfX1AbWVkaWEgb25seSBzY3JlZW4gYW5kICgtd2Via2l0LW1pbi1kZXZpY2UtcGl4ZWwtcmF0aW86Mil7I2xvZ297YmFja2dyb3VuZDp1cmwoLy93d3cuZ29vZ2xlLmNvbS9pbWFnZXMvbG9nb3MvZXJyb3JwYWdlL2Vycm9yX2xvZ28tMTUweDU0LTJ4LnBuZykgbm8tcmVwZWF0Oy13ZWJraXQtYmFja2dyb3VuZC1zaXplOjEwMCUgMTAwJX19I2xvZ297ZGlzcGxheTppbmxpbmUtYmxvY2s7aGVpZ2h0OjU0cHg7d2lkdGg6MTUwcHh9CiAgPC9zdHlsZT4KICA8YSBocmVmPS8vd3d3Lmdvb2dsZS5jb20vPjxzcGFuIGlkPWxvZ28gYXJpYS1sYWJlbD1Hb29nbGU+PC9zcGFuPjwvYT4KICA8cD48Yj40MDQuPC9iPiA8aW5zPlRoYXTigJlzIGFuIGVycm9yLjwvaW5zPgogIDxwPiAgPGlucz5UaGF04oCZcyBhbGwgd2Uga25vdy48L2lucz4K", "ok": false, "headers": [["content-length", "1449"], ["content-type", "text/html; charset=utf-8"]], "status": 404, "status_text": "Not Found"}}, "base_uri": "https://localhost:8080/", "height": 521, "output_embedded_package_id": "1JBILJemJOX0MmjyS1PzZWctK2RfTmc2S"} id="cuB8nG5u9jxi" executionInfo={"status": "ok", "timestamp": 1673879005435, "user_tz": 180, "elapsed": 25460, "user": {"displayName": "Roberto Federico Ausas", "userId": "01910242568345374894"}} outputId="1cc87655-b567-41b8-cde3-93192a979b46"
plot_mesh(msh, "mesh.html")
IPython.display.HTML(filename="mesh.html")
