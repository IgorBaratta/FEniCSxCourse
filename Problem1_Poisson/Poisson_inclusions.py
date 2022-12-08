import gmsh
import numpy as np
#import pyvista

from dolfinx.fem import (Constant, dirichletbc, Function, FunctionSpace, assemble_scalar,
                         form, locate_dofs_geometrical, locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_unit_square, locate_entities
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

inclusion_marker = 3
background_marker = 2

ninclusions = 3
Lx = 2.0
Ly = 2.0
R1 = 0.25
R2 = 0.15
R3 = 0.25

#--------------------------------------------------------------------
#--- Preprocess: Mesh generation, boundary and region identification

import meshio
def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:,:2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
    return out_mesh

def GenerateMesh():

    gmsh.initialize()
    proc = MPI.COMM_WORLD.rank

    mass1 = np.pi*R1**2
    mass2 = np.pi*R2**2
    mass3 = np.pi*R3**2
    mass_inc = mass1 + mass2 + mass3

    if proc == 0:
        # We create one rectangle and the circular inclusion
        background = gmsh.model.occ.addRectangle(0, 0, 0, Lx, Ly)
        inclusion1 = gmsh.model.occ.addDisk(0.5, 1.0, 0, R1, R1)
        inclusion2 = gmsh.model.occ.addDisk(1.0, 1.5, 0, R2, R2)
        inclusion3 = gmsh.model.occ.addDisk(1.5, 1.0, 0, R3, R3)
        gmsh.model.occ.synchronize()
        all_inclusions = [(2, inclusion1)]
        all_inclusions.extend([(2, inclusion2)])
        all_inclusions.extend([(2, inclusion3)])
        whole_domain = gmsh.model.occ.fragment([(2, background)], all_inclusions)
        gmsh.model.occ.synchronize()

        background_surfaces = []
        other_surfaces = []
        for domain in whole_domain[0]:
            com = gmsh.model.occ.getCenterOfMass(domain[0], domain[1])
            mass = gmsh.model.occ.getMass(domain[0], domain[1])
            print(mass, com)
            # Identify the square by its mass
            if np.isclose(mass, (Lx*Ly - mass_inc)):
                gmsh.model.addPhysicalGroup(domain[0], [domain[1]], tag=background_marker)
                background_surfaces.append(domain)
            elif np.isclose(np.linalg.norm(com), np.sqrt((0.5)**2 + (1.0)**2)):
                gmsh.model.addPhysicalGroup(domain[0], [domain[1]], tag=inclusion_marker)
                other_surfaces.append(domain)
            elif np.isclose(np.linalg.norm(com), np.sqrt((1.0)**2 + (1.5)**2)) and com[1] > 1.0:
                gmsh.model.addPhysicalGroup(domain[0], [domain[1]], tag=inclusion_marker+1)
                other_surfaces.append(domain)
            elif np.isclose(np.linalg.norm(com), np.sqrt((1.5)**2 + (1.0)**2)):
                gmsh.model.addPhysicalGroup(domain[0], [domain[1]], tag=inclusion_marker+2)
                other_surfaces.append(domain)
    
        # Tag the left and right boundaries
        left = []
        right = []
        for line in gmsh.model.getEntities(dim=1):
            com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
            if np.isclose(com[0], 0.0):
                print('L', line)
                left.append(line[1])
            if np.isclose(com[0], Lx):
                print('R', line)
                right.append(line[1])
        gmsh.model.addPhysicalGroup(1, left, left_wall)
        gmsh.model.addPhysicalGroup(1, right, right_wall)

    
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.0075)
    gmsh.model.mesh.generate(2)    
    gmsh.write("mesh.msh")
    gmsh.finalize()

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

    return mesh, ct, ft
#--------------------------------------------------------------------

mesh, ct, ft = GenerateMesh()

Q = FunctionSpace(mesh, ("DG", 0))
mu = Function(Q)

muA = 1.0
muB = 10000.0
uleft = 100.0
uright = 1.0


background_cells = ct.indices[ct.values==background_marker]
mu.x.array[background_cells] = np.full(len(background_cells), muA)

inclusion_cells = ct.indices[ct.values >= inclusion_marker]
mu.x.array[inclusion_cells]  = np.full(len(inclusion_cells), muB)

V = FunctionSpace(mesh, ("CG", 1))
u_bc = Function(V)

# Boundary conditions
# Identify facets with Dirichlet bcs
left_facets = ft.indices[ft.values==left_wall]
right_facets = ft.indices[ft.values==right_wall]

# Identify the associated unknowns
left_dofs = locate_dofs_topological(V, mesh.topology.dim-1, left_facets)
right_dofs = locate_dofs_topological(V, mesh.topology.dim-1, right_facets)

# Set Dirchlet values
bcs = [dirichletbc(ScalarType(uleft), left_dofs, V), dirichletbc(ScalarType(uright), right_dofs, V)]

# Variational formulation
u, v = TrialFunction(V), TestFunction(V)
a = inner(mu*grad(u), grad(v)) * dx
x = SpatialCoordinate(mesh)
source = Constant(mesh, ScalarType(0))
L = source * v * dx

# Solve the problem
problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()
uh.name = "Temperature"

ue = uleft - (uleft - uright)*x[0]/Lx
gradue = as_vector([-(uleft - uright), 0.0])
errorL2 = assemble_scalar(form((uh - ue)**2*dx))
errorH1 = errorL2 + assemble_scalar(form((grad(uh) - gradue)**2*dx))
print("    |-L2error=", np.sqrt(errorL2))
print("    |-H1error=", np.sqrt(errorH1))

# Compute effective mu
n = FacetNormal(mesh)
ds = Measure('ds', subdomain_data=ft)
Qleft  = assemble_scalar(form(inner(mu*grad(uh),n) * ds(3)))
Qright = assemble_scalar(form(inner(mu*grad(uh),n) * ds(1)))
gen = assemble_scalar(form(source * dx))
print(Qleft, Qright, gen)

# Compute mean temperature in the inclusions
dx = Measure('dx', subdomain_data=ct)
one = Constant(mesh, ScalarType(1))
Tincav = []
for k in range(ninclusions):
    area = assemble_scalar(form(one * dx(inclusion_marker+k)))
    Tinc = assemble_scalar(form(uh  * dx(inclusion_marker+k)))
    Tincav.append(Tinc/area)
    print(Tincav[k])
    
# Save the results
with XDMFFile(MPI.COMM_WORLD, "temperature.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(uh)
