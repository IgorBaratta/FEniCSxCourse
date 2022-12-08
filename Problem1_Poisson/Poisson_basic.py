import numpy as np

from dolfinx.fem import (Constant, dirichletbc, Function, FunctionSpace, assemble_scalar,
                         form, locate_dofs_geometrical, locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_rectangle, CellType, locate_entities
from dolfinx.plot import create_vtk_mesh

from ufl import (SpatialCoordinate, TestFunction, TrialFunction, Measure,
                 dx, ds, grad, inner, FacetNormal, as_vector)

from mpi4py import MPI
from petsc4py.PETSc import ScalarType

# User defined data
bottom_wall = 0
right_wall  = 1
top_wall    = 2
left_wall   = 3

#--------------------------------------------------------------------
#--- Preprocess: Mesh generation and boundary identification

Lx, Ly = 1.0, 1.0
msh = create_rectangle(comm=MPI.COMM_WORLD,
                       points=((0.0, 0.0), (Lx, Ly)), n=(64, 64),
                       cell_type=CellType.triangle)
                            
tdim = msh.topology.dim
fdim = tdim - 1
msh.topology.create_connectivity(fdim, tdim)

V = FunctionSpace(msh, ("CG", 1))

# Boundary condition
uleft = 100.0
uright = 1.0

dofsL = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0))
dofsR = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], Lx))
                            
# Set Dirchlet values
bcs = [dirichletbc(ScalarType(uleft), dofsL, V), dirichletbc(ScalarType(uright), dofsR, V)]

# Variational formulation
kappa = Constant(msh, ScalarType(0.1))
u, v = TrialFunction(V), TestFunction(V)
a = inner(kappa*grad(u), grad(v)) * dx
source = Constant(msh, ScalarType(10))
L = source * v * dx

# Solve the problem
problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()
uh.name = "Temperature"

x = SpatialCoordinate(msh)
ue = uleft - (uleft - uright)*x[0]/Lx
gradue = as_vector([-(uleft - uright), 0.0])
errorL2 = assemble_scalar(form((uh - ue)**2*dx))
errorH1 = errorL2 + assemble_scalar(form((grad(uh) - gradue)**2*dx))
print("    |-L2error=", np.sqrt(errorL2))
print("    |-H1error=", np.sqrt(errorH1))
    
# Save the results
with XDMFFile(MPI.COMM_WORLD, "temperature.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh)
    xdmf.write_function(uh)
