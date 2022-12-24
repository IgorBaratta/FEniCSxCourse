import gmsh
from dolfinx import fem, io, mesh
from dolfinx.io import XDMFFile

import ufl
from ufl import dx, grad, inner
import numpy as np
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

if(False):
    msh = mesh.create_rectangle(comm=MPI.COMM_WORLD,
                                points=((0.0, 0.0), (1.0, 1.0)), n=(32, 32),
                                cell_type=mesh.CellType.triangle)

import meshio
def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:,:2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
    return out_mesh

gmsh.initialize()

proc = MPI.COMM_WORLD.rank
if proc == 0:

    lc = 0.05

    Db   = 0.4
    Hb   = 0.4
    Hp   = 6*Hb
    R    = 3*Hb
    TT   = np.sqrt(R*R - 4*Hb*Hb)
    print(TT)
    
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
    gmsh.model.setPhysicalName(2, ps, "My surface") 
    
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)

    gmsh.write("./corpo.msh")
    gmsh.finalize()

    # Read in mesh
    msh = meshio.read("./corpo.msh")
    
    # Create and save one file for the mesh, and one file for the facets 
    triangle_mesh = create_mesh(msh, "triangle", prune_z=True)
    line_mesh = create_mesh(msh, "line", prune_z=True)
    meshio.write("corpo.xdmf", triangle_mesh)
    meshio.write("mt.xdmf", line_mesh)
    
    with XDMFFile(MPI.COMM_WORLD, "corpo.xdmf", "r") as xdmf:
        msh = xdmf.read_mesh(name="Grid")
        ct = xdmf.read_meshtags(msh, name="Grid")
        msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim-1)
    with XDMFFile(MPI.COMM_WORLD, "mt.xdmf", "r") as xdmf:
        ft = xdmf.read_meshtags(msh, name="Grid")

Vd = fem.FunctionSpace(msh, ("Lagrange", 1))

tdim = msh.topology.dim
fdim = tdim - 1
msh.topology.create_connectivity(fdim, tdim)
facets = np.flatnonzero(mesh.compute_boundary_facets(msh.topology))
dofs = fem.locate_dofs_topological(Vd, fdim, facets)
u_boundary = fem.Constant(msh, ScalarType(0.0))
bc = fem.dirichletbc(u_boundary, dofs, Vd)

u  = ufl.TrialFunction(Vd)
v  = ufl.TestFunction(Vd)
x  = ufl.SpatialCoordinate(msh)
f  = fem.Constant(msh, ScalarType(1.0))
mu = fem.Constant(msh, ScalarType(1.0))
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
