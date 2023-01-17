import numpy as np
import dolfinx
import ufl
from mpi4py import MPI
from dolfinx.io import XDMFFile

p = 2
k0 = 10 * np.pi

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 100, 100)
element = ufl.FiniteElement("Lagrange", ufl.triangle, p)
n = ufl.FacetNormal(mesh)

# Definition of function space
V = dolfinx.fem.FunctionSpace(mesh, element)


def incoming_wave(x):
    d = np.cos(theta) * x[0] + np.sin(theta) * x[1]
    return np.exp(1.0j * k0 * d)


# Incoming wave
theta = np.pi/4
ui = dolfinx.fem.Function(V)
ui.interpolate(incoming_wave)
g = ufl.dot(ufl.grad(ui), n) + 1j * k0 * ui

# Define variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Weak Form
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
    - k0**2 * ufl.inner(u, v) * ufl.dx \
    + 1j * k0 * ufl.inner(u, v) * ufl.ds
L = ufl.inner(g, v) * ufl.ds

opt = {"ksp_type": "preonly", "pc_type": "lu"}
problem = dolfinx.fem.petsc.LinearProblem(a, L, petsc_options=opt)
uh = problem.solve()
uh.name = "u"

with XDMFFile(MPI.COMM_WORLD, "out.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(u)
