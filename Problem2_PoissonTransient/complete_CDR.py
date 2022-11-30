import gmsh
import numpy as np

from dolfinx.fem import (Constant, dirichletbc, Function, FunctionSpace, assemble_scalar,
                         form, locate_dofs_geometrical, locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_rectangle, locate_entities, CellType
from dolfinx.plot import create_vtk_mesh

from ufl import (SpatialCoordinate, TestFunction, TrialFunction,
                 dx, grad, div, inner, dot, as_vector, Circumradius, sqrt)

from mpi4py import MPI
from petsc4py.PETSc import ScalarType


#--------------------------------------------------------------------
#--- Preprocess: Mesh generation, boundary and region identification


#celltype = mesh.CellType.quadrilateral
celltype =CellType.triangle
Lx = 2.0
Ly = 1.0

flag_unstructured = False
flag_refine_exit = True

if(flag_unstructured == False):
    mesh = create_rectangle(comm=MPI.COMM_WORLD,
                            points=((0.0, 0.0), (Lx, Ly)), n=(64, 64),
                            cell_type=celltype)

    def Omega_0(x):
        return x[0] <= 1.0

    def Omega_1(x):
        return x[0] >= 1.0

    cells_0 = locate_entities(mesh, mesh.topology.dim, Omega_0)
    cells_1 = locate_entities(mesh, mesh.topology.dim, Omega_1)
    
else:
    gmsh.initialize()
    proc = MPI.COMM_WORLD.rank 
    leftdom_marker = 2
    rightdom_marker = 1
    leftwall_marker = 1
    rightwall_marker = 0
    if proc == 0:
        # We create one rectangle for each subdomain
        gmsh.model.occ.addRectangle(0, 0, 0, Lx/2, Ly, tag=1)
        gmsh.model.occ.addRectangle(Lx/2, 0, 0, Lx/2, Ly, tag=2)
        # We fuse the two rectangles and keep the interface between them 
        whole_domain = gmsh.model.occ.fragment([(2,1)],[(2,2)])
        gmsh.model.occ.synchronize()
   
        # Mark the left and right subdomains
        leftdom, rightdom = [], [] # None, None
        for surface in whole_domain[0]:
            com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
            if np.allclose(com, [Lx/4,Ly/2, 0]):
                leftdom.append(surface)
                gmsh.model.addPhysicalGroup(2, [surface[1]], leftdom_marker)
            else:
                rightdom.append(surface)
                gmsh.model.addPhysicalGroup(2, [surface[1]], rightdom_marker)

        # Tag the left and right boundaries
        leftwall, rightwall = [], []
        for line in gmsh.model.getEntities(dim=1):
            com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
            if np.isclose(com[0], 0):
                leftwall.append(line[1])
            if np.isclose(com[0], Lx):
                rightwall.append(line[1])
        gmsh.model.addPhysicalGroup(1, leftwall, leftwall_marker)
        gmsh.model.addPhysicalGroup(1, rightwall, rightwall_marker)

        if(flag_refine_exit):
            gmsh.model.mesh.field.add("Distance", 1)
            gmsh.model.mesh.field.setNumbers(1, "EdgesList", rightwall)
            gmsh.model.mesh.field.add("Threshold", 2)
            gmsh.model.mesh.field.setNumber(2, "IField", 1)
            r = 0.1
            gmsh.model.mesh.field.setNumber(2, "LcMin", r / 2)
            gmsh.model.mesh.field.setNumber(2, "LcMax", 1.5 * r)
            gmsh.model.mesh.field.setNumber(2, "DistMin", 2 * r)
            gmsh.model.mesh.field.setNumber(2, "DistMax", 4 * r)
            gmsh.model.mesh.field.setAsBackgroundMesh(2)
        else:
            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.01)
            
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(2)
        gmsh.write("mesh.msh")
    gmsh.finalize()

    import meshio
    def create_mesh(mesh, cell_type, prune_z=False):
        cells = mesh.get_cells_type(cell_type)
        cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
        points = mesh.points[:,:2] if prune_z else mesh.points
        out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
        return out_mesh

    if proc == 0:
        # Read in mesh
        msh = meshio.read("mesh.msh")
   
        # Create and save one file for the mesh, and one file for the facets 
        triangle_mesh = create_mesh(msh, "triangle", prune_z=True)
        line_mesh = create_mesh(msh, "line", prune_z=True)
        meshio.write("mesh.xdmf", triangle_mesh)
        meshio.write("mt.xdmf", line_mesh)

    with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        ct = xdmf.read_meshtags(mesh, name="Grid")
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim-1)
    with XDMFFile(MPI.COMM_WORLD, "mt.xdmf", "r") as xdmf:
        ft = xdmf.read_meshtags(mesh, name="Grid")

    cells_0 = ct.indices[ct.values == leftdom_marker]
    cells_1 = ct.indices[ct.values == rightdom_marker]
    
Q = FunctionSpace(mesh, ("DG", 0))

#--- End preprocessing
#-------------------------------------------------

#-------------------------------------------------
# CDR problem

kappa = Constant(mesh, ScalarType(0.1))
sigma = 0.0
sigmaC = Constant(mesh, ScalarType(sigma))
source =  Function(Q)
beta = as_vector([1.0, 0.0])

#icase = 'REAC_BL'
icase = 'CONV_BL'
#iestab = 'SUPG'
iestab = 'NONE'

V = FunctionSpace(mesh, ("CG", 1))
u, v = TrialFunction(V), TestFunction(V)
a  = inner(kappa*grad(u), grad(v)) * dx

bcs = []
if  (icase == 'REAC_BL'):
    a = a + sigmaC*inner(u, v) * dx
    source.x.array[cells_0] = np.full_like(cells_0, 0.0, dtype=ScalarType)
    source.x.array[cells_1] = np.full_like(cells_1, sigma, dtype=ScalarType)
    L = source * v * dx
elif(icase == 'CONV_BL'):
    a = a + dot(beta,grad(u))*v*dx
    source.x.array[cells_0] = np.full_like(cells_0, 0.0, dtype=ScalarType)
    source.x.array[cells_1] = np.full_like(cells_1, 0.0, dtype=ScalarType)
    L = source * v * dx
    x = SpatialCoordinate(mesh)
    dofsL = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0))
    dofsR = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], Lx))
    bcs = [dirichletbc(ScalarType(0), dofsL, V), dirichletbc(ScalarType(1), dofsR, V)]
    
if(iestab == 'SUPG'):
    hk = 2 * Circumradius(mesh)
    tauk = 1.0/(4*kappa/hk**2 + 2.0*sqrt(dot(beta,beta))/hk + sigma)
    lhs_eq = -div(kappa*grad(u)) + dot(beta,grad(u)) + sigma*u
    rhs_eq = source
    a = a + tauk * lhs_eq * dot(beta,grad(v)) * dx
    L = L + tauk * rhs_eq * dot(beta,grad(v)) * dx

# Solve the problem
problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()
uh.name = "Temperature"

# Save the results
with XDMFFile(MPI.COMM_WORLD, "temperature.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(uh)



    
