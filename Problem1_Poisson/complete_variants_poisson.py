import gmsh
import numpy as np
#import pyvista

from dolfinx.fem import (Constant, dirichletbc, Function, FunctionSpace, assemble_scalar,
                         form, locate_dofs_geometrical, locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_unit_square, locate_entities
from dolfinx.plot import create_vtk_mesh

from ufl import (SpatialCoordinate, TestFunction, TrialFunction, Measure, sin,
                 dx, ds, grad, inner, FacetNormal, as_vector)

from mpi4py import MPI
from petsc4py.PETSc import ScalarType

# User defined data
bottom_wall = 0
right_wall  = 1
top_wall    = 2
left_wall   = 3
hole1_wall  = 4
hole2_wall  = 5
hole3_wall  = 6

inclusion_marker = 3
background_marker = 2

ninclusions = 3
Lx = 2.0
Ly = 2.0
R1 = 0.25
R2 = 0.15
R3 = 0.25

gdim = 2

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

    if proc == 0:
        # We create one rectangle and the circular inclusion
        rectangle = gmsh.model.occ.addRectangle(0, 0, 0, Lx, Ly)
        hole1 = gmsh.model.occ.addDisk(0.5, 1.0, 0, R1, R1)
        hole2 = gmsh.model.occ.addDisk(1.0, 1.5, 0, R2, R2)
        hole3 = gmsh.model.occ.addDisk(1.5, 1.0, 0, R3, R3)
        gmsh.model.occ.synchronize()
        all_holes = [(2, hole1)]
        all_holes.extend([(2, hole2)])
        all_holes.extend([(2, hole3)])
        whole_domain = gmsh.model.occ.cut([(gdim, rectangle)], all_holes)
        gmsh.model.occ.synchronize()
        background_surfaces = []
        for domain in whole_domain[0]:
            gmsh.model.addPhysicalGroup(domain[0], [domain[1]], tag=background_marker)
            background_surfaces.append(domain)
        
        # Tag the the different boundaries
        left = []
        right = []
        top = []
        bottom = []
        hole1,hole2,hole3 = [], [], []
        for line in gmsh.model.getEntities(dim=1):
            com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
            if np.isclose(com[0], 0.0):
                #print('L', line, com)
                left.append(line[1])
            if np.isclose(com[0], Lx):
                #print('R', line, com)
                right.append(line[1])
            if np.isclose(com[1], 0.0):
                #print('B', line, com)
                bottom.append(line[1])
            if np.isclose(com[1], Ly):
                #print('T', line, com)
                top.append(line[1])
            if np.isclose(np.linalg.norm(com), np.sqrt((0.5)**2 + (1.0)**2)):
                #print('H1', line, com)
                hole1.append(line[1])
            elif np.isclose(np.linalg.norm(com), np.sqrt((1.0)**2 + (1.5)**2)) and np.isclose(com[1], 1.5):
                #print('H2', line, com)
                hole2.append(line[1])
            elif np.isclose(np.linalg.norm(com), np.sqrt((1.5)**2 + (1.0)**2)):
                #print('H3', line, com)
                hole3.append(line[1])
                
        gmsh.model.addPhysicalGroup(1, left, left_wall)
        gmsh.model.addPhysicalGroup(1, right, right_wall)
        gmsh.model.addPhysicalGroup(1, top, top_wall)
        gmsh.model.addPhysicalGroup(1, bottom, bottom_wall)
        gmsh.model.addPhysicalGroup(1, hole1, hole1_wall)
        gmsh.model.addPhysicalGroup(1, hole2, hole2_wall)
        gmsh.model.addPhysicalGroup(1, hole3, hole3_wall)
        gmsh.model.occ.synchronize()

    if(True):
        r = 0.01
        res_min = r
        res_max = 4 * r
        if proc == 0:
            gmsh.model.mesh.field.add("Distance", 1)
            gmsh.model.mesh.field.setNumbers(1, "EdgesList", hole1+hole2+hole3)
            gmsh.model.mesh.field.add("Threshold", 2)
            gmsh.model.mesh.field.setNumber(2, "IField", 1)
            gmsh.model.mesh.field.setNumber(2, "LcMin", res_min)
            gmsh.model.mesh.field.setNumber(2, "LcMax", res_max)
            gmsh.model.mesh.field.setNumber(2, "DistMin", 2*r)
            gmsh.model.mesh.field.setNumber(2, "DistMax", 4*r)
            # We take the minimum of the two fields as the mesh size
            gmsh.model.mesh.field.add("Min", 5)
            gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2])
            gmsh.model.mesh.field.setAsBackgroundMesh(2)
            # Generate mesh
            gmsh.option.setNumber("Mesh.Algorithm", 6)
            gmsh.model.mesh.generate(2)
    else:
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.1)
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

# Material parameters and boundary values
mu = Constant(mesh, ScalarType(1.0))
gamma = Constant(mesh, ScalarType(1000.0))
source = Constant(mesh, ScalarType(0.0))
uleft = 100.0
uright = 10.0
T1 = 74.41042294352245
T2 = 50.49999815995027
T3 = 26.589580423425417
ht = Constant(mesh, ScalarType(1000.0))
hb = Constant(mesh, ScalarType(1000.0))

V = FunctionSpace(mesh, ("CG", 1))

# Boundary conditions
# Identify facets with Dirichlet bcs
left_facets  = ft.indices[ft.values==left_wall]
right_facets = ft.indices[ft.values==right_wall]
hole1_facets = ft.indices[ft.values==hole1_wall]
hole2_facets = ft.indices[ft.values==hole2_wall]
hole3_facets = ft.indices[ft.values==hole3_wall]

# Identify the associated unknowns
left_dofs = locate_dofs_topological(V, mesh.topology.dim-1, left_facets)
right_dofs = locate_dofs_topological(V, mesh.topology.dim-1, right_facets)
hole1_dofs = locate_dofs_topological(V, mesh.topology.dim-1, hole1_facets)
hole2_dofs = locate_dofs_topological(V, mesh.topology.dim-1, hole2_facets)
hole3_dofs = locate_dofs_topological(V, mesh.topology.dim-1, hole3_facets)

# Set Dirchlet values
bcs = []
bcs.append(dirichletbc(ScalarType(uleft), left_dofs, V))
bcs.append(dirichletbc(ScalarType(uright), right_dofs, V))
bcs.append(dirichletbc(ScalarType(T1), hole1_dofs, V))
bcs.append(dirichletbc(ScalarType(T2), hole2_dofs, V))
bcs.append(dirichletbc(ScalarType(T3), hole3_dofs, V))

ds = Measure('ds', subdomain_data=ft)
one = Constant(mesh, ScalarType(1))
aux = assemble_scalar(form(one * (ds(0)+ds(2))))
print(aux)

# Variational formulation
x = SpatialCoordinate(mesh) 
u, v = TrialFunction(V), TestFunction(V)
a = inner(mu*grad(u), grad(v))*dx + inner(gamma*u,v)*(ds(0) + ds(2))
L = source * v * dx + inner(hb,v)*ds(0) + inner(ht,v)*ds(2)

# Solve the problem
problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()
uh.name = "Temperature"

# Save the results
with XDMFFile(MPI.COMM_WORLD, "temperature.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(uh)

xdmf_evol = XDMFFile(MPI.COMM_WORLD, "temperature_evol.xdmf", "w")
xdmf_evol.write_mesh(mesh)

if(False):
    for gam in [0.00001, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        print('Solving for gamma=', gam)
        gamma = Constant(mesh, ScalarType(gam))
        ht = Constant(mesh, ScalarType(10.0*gam))
        hb = Constant(mesh, ScalarType(10.0*gam))
        a = inner(mu*grad(u), grad(v))*dx + inner(gamma*u,v)*(ds(0) + ds(2))
        L = source * v * dx + inner(hb,v)*ds(0) + inner(ht,v)*ds(2)
        problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        uh = problem.solve()
        uh.name = "Temperature"
        xdmf_evol.write_function(uh, gam)

if(True):   
    def gfunc1(x):
        theta = np.arctan2( x[1]-1.0, x[0]-0.5 )
        return 0.1*(np.sin(2*theta))**2
    def gfunc2(x):
        theta = np.arctan2( x[1]-1.5, x[0]-1.0 )
        return 0.1*(np.sin(3*theta))**2
    def gfunc3(x):
        theta = np.arctan2( x[1]-1.0, x[0]-1.5 )
        return 0.1*(np.sin(4*theta))**2

    def hfunc(x):
        return 0.2*x[0]*(Lx - x[0])

    uleft = 0.0
    uright = 0.0

    gamma = Constant(mesh, ScalarType(0.0))
    source = Constant(mesh, ScalarType(0.0))
    
    u_h1 = Function(V)
    u_h1.interpolate(gfunc1)
    u_h2 = Function(V)
    u_h2.interpolate(gfunc2)
    u_h3 = Function(V)
    u_h3.interpolate(gfunc3)

    bcs = []
    bcs.append(dirichletbc(ScalarType(uleft), left_dofs, V))
    bcs.append(dirichletbc(ScalarType(uright), right_dofs, V))
    bcs.append(dirichletbc(u_h1, hole1_dofs))
    bcs.append(dirichletbc(u_h2, hole2_dofs))
    bcs.append(dirichletbc(u_h3, hole3_dofs))

    a = inner(mu*grad(u), grad(v))*dx + inner(gamma*u,v)*(ds(0) + ds(2))
    L = source * v * dx + inner(hfunc(x),v)*ds(0) + inner(hfunc(x),v)*ds(2)
    problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()
    uh.name = "Temperature"

    xdmf_gh = XDMFFile(MPI.COMM_WORLD, "temperature_gh.xdmf", "w")
    xdmf_gh.write_mesh(mesh)
    xdmf_gh.write_function(uh)
    
