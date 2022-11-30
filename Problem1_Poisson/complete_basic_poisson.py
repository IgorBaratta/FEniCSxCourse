from dolfinx import fem, io, mesh, cpp
import ufl
from ufl import dx, ds, grad, inner, dot, FacetNormal
import numpy as np
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

msh = mesh.create_rectangle(comm=MPI.COMM_WORLD,
                            points=((0.0, 0.0), (1.0, 1.0)), n=(64, 64),
                            cell_type=mesh.CellType.triangle)

V = fem.FunctionSpace(msh, ("Lagrange", 1))

tdim = msh.topology.dim
fdim = tdim - 1
msh.topology.create_connectivity(fdim, tdim)
facets = np.flatnonzero(mesh.compute_boundary_facets(msh.topology))
dofs = fem.locate_dofs_topological(V, fdim, facets)
u_boundary = fem.Constant(msh, ScalarType(0.0))
bc = fem.dirichletbc(u_boundary, dofs, V)

u  = ufl.TrialFunction(V)
v  = ufl.TestFunction(V)
x  = ufl.SpatialCoordinate(msh)
f  = 1.0 #fem.Constant(msh, ScalarType(1.0))
mu = fem.Constant(msh, ScalarType(1.0))
#mu = 0.01 + ufl.exp(-100 * ((x[0]-0.5)**2 + (x[1] - 0.5)**2))
a  = inner(mu * grad(u), grad(v)) * dx
L  = inner(f, v) * dx

one = fem.Constant(msh, ScalarType(1))
area = fem.form(one * dx)
print(fem.assemble_scalar(area))

opts={"ksp_type": "preonly", "pc_type": "lu"}
problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options=opts)
uh = problem.solve()
uh.name = "u"

with io.XDMFFile(msh.comm, "poisson.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(uh)
    
tau = mu*ufl.grad(uh)
magtau = ufl.sqrt(dot(tau, tau))
Vstress = fem.FunctionSpace(msh, ("DG", 0))
stress_expr = fem.Expression(magtau, Vstress.element.interpolation_points)
stresses = fem.Function(Vstress)
stresses.interpolate(stress_expr)
with io.XDMFFile(msh.comm, "stress.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(stresses)


Vstress_v = fem.VectorFunctionSpace(msh, ("DG", 0))
stress_expr_v = fem.Expression(tau, Vstress_v.element.interpolation_points)
stresses_v = fem.Function(Vstress_v)
stresses_v.interpolate(stress_expr_v)
with io.XDMFFile(msh.comm, "stress_v.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(stresses_v)

n = FacetNormal(msh)    
wss = fem.form(ufl.inner(tau,n)*ds)
print(fem.assemble_scalar(wss))


import gmsh
gmsh.initialize()


membrane = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
gmsh.model.occ.synchronize()

gdim = 2
status = gmsh.model.addPhysicalGroup(gdim, [membrane], 1)

gmsh.option.setNumber("Mesh.CharacteristicLengthMin",0.05)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax",0.05)
gmsh.model.mesh.generate(gdim)

if MPI.COMM_WORLD.rank == 0:
    # Get mesh geometry
    geometry_data = io.extract_gmsh_geometry(gmsh.model)
    # Get mesh topology for each element
    topology_data = io.extract_gmsh_topology_and_markers(gmsh.model)

if MPI.COMM_WORLD.rank == 0:
    # Extract the cell type and number of nodes per cell and broadcast
    # it to the other processors 
    gmsh_cell_type = list(topology_data.keys())[0]    
    properties = gmsh.model.mesh.getElementProperties(gmsh_cell_type)
    name, dim, order, num_nodes, local_coords, _ = properties
    cells = topology_data[gmsh_cell_type]["topology"]
    cell_id, num_nodes = MPI.COMM_WORLD.bcast([gmsh_cell_type, num_nodes], root=0)
else:        
    cell_id, num_nodes = MPI.COMM_WORLD.bcast([None, None], root=0)
    cells, geometry_data = np.empty([0, num_nodes]), np.empty([0, gdim])

# Permute topology data from MSH-ordering to dolfinx-ordering
ufl_domain = io.ufl_mesh_from_gmsh(cell_id, gdim)
gmsh_cell_perm = io.cell_perm_gmsh(cpp.mesh.to_type(str(ufl_domain.ufl_cell())), num_nodes)
cells = cells[:, gmsh_cell_perm]

# Create distributed mesh
domain = mesh.create_mesh(MPI.COMM_WORLD, cells, geometry_data[:, :gdim], ufl_domain)

V = fem.FunctionSpace(domain, ("CG", 1))

x = ufl.SpatialCoordinate(domain)
beta = fem.Constant(domain, ScalarType(12))
R0 = fem.Constant(domain, ScalarType(0.3))
p = 4 * ufl.exp(-beta**2 * (x[0]**2 + (x[1] - R0)**2))

def on_boundary(x):
    return np.isclose(np.sqrt(x[0]**2 + x[1]**2), 1)
boundary_dofs = fem.locate_dofs_geometrical(V, on_boundary)

bc = fem.dirichletbc(ScalarType(0), boundary_dofs, V)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = p * v * ufl.dx
problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

Q = fem.FunctionSpace(domain, ("CG", 5))
expr = fem.Expression(p, Q.element.interpolation_points)
pressure = fem.Function(Q)
pressure.interpolate(expr)

pressure.name = "Load"
uh.name = "Deflection"
with io.XDMFFile(MPI.COMM_WORLD, "results_membrane.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(pressure)
    xdmf.write_function(uh)
