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
    
    gmsh.model.occ.addPoint(0, 0, 0, lc, 1)
    gmsh.model.occ.addPoint(Db, 0, 0, lc, 2)
    gmsh.model.occ.addPoint(Db, Hb, 0, 0.5*lc, 3)
    gmsh.model.occ.addPoint(TT+Db, 3*Hb, 0, lc, 4)
    gmsh.model.occ.addPoint(Db, 5*Hb, 0, lc, 5)
    gmsh.model.occ.addPoint(Db, 6*Hb, 0, 0.5*lc, 6)
    gmsh.model.occ.addPoint(0, 6*Hb, 0, lc, 7)
    gmsh.model.occ.addPoint(0, 3*Hb, 0, 0.1*lc, 8)
    gmsh.model.occ.addPoint(TT+Db-R, 3*Hb, 0, 0.1*lc, 9)
    
    gmsh.model.occ.addLine(1, 2, 1)
    gmsh.model.occ.addLine(2, 3, 2)
    gmsh.model.occ.addCircleArc(3, 4, 9, 3)
    gmsh.model.occ.addCircleArc(9, 4, 5, 4)
    gmsh.model.occ.addLine(5, 6, 5)
    gmsh.model.occ.addLine(6, 7, 6)
    gmsh.model.occ.addLine(7, 8, 7)
    gmsh.model.occ.addLine(8, 1, 8)
    
    gmsh.model.occ.addCurveLoop([1, 2, 3, 4, 5, 6, 7, 8], 1)
    gmsh.model.occ.addPlaneSurface([1], 1)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(1, [1, 2, 3, 4, 5, 6, 7, 8], 101)
    ps = gmsh.model.addPhysicalGroup(2, [1])
    gmsh.model.setPhysicalName(2, ps, "My surface 1")

    gmsh.model.occ.addPoint(-Db, 0, 0, lc, 10)
    gmsh.model.occ.addPoint(-Db, Hb, 0, 0.5*lc, 11)
    gmsh.model.occ.addPoint(-(TT+Db), 3*Hb, 0, lc, 12)
    gmsh.model.occ.addPoint(-Db, 5*Hb, 0, lc, 13)
    gmsh.model.occ.addPoint(-Db, 6*Hb, 0, 0.5*lc, 14)
    gmsh.model.occ.addPoint(-(TT+Db-R), 3*Hb, 0, 0.1*lc, 15)
    
    gmsh.model.occ.addLine(1, 8, 9)
    gmsh.model.occ.addLine(8, 7, 10)
    gmsh.model.occ.addLine(7, 14, 11)
    gmsh.model.occ.addLine(14, 13, 12)
    gmsh.model.occ.addCircleArc(13, 12, 15, 13)
    gmsh.model.occ.addCircleArc(15, 12, 11, 14)
    gmsh.model.occ.addLine(11, 10, 15)
    gmsh.model.occ.addLine(10, 1, 16)
    
    gmsh.model.occ.addCurveLoop([9, 10, 11, 12, 13, 14, 15, 16], 2)
    gmsh.model.occ.addPlaneSurface([2], 2)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(1, [9, 10, 11, 12, 13, 14, 15, 16], 103)
    ps = gmsh.model.addPhysicalGroup(2, [2])
    gmsh.model.setPhysicalName(2, ps, "My surface 2")
    gmsh.model.occ.synchronize()

    ov1 = gmsh.model.occ.revolve([(2, 1)], 0, 0, 0, 0, 1, 0, 1.5*np.pi)
    #ov2 = gmsh.model.occ.revolve([(2, 1)], 0, 0, 0, 0, 1, 0,  np.pi / 2)
    #ov3 = gmsh.model.occ.revolve([(2, 2)], 0, 0, 0, 0, 1, 0, -np.pi / 2)
    #ov4 = gmsh.model.occ.revolve([(2, 2)], 0, 0, 0, 0, 1, 0,  np.pi / 2)
    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(3, [ov1[1][1]], 105)
    #gmsh.model.addPhysicalGroup(3, [ov2[1][1]], 106)
    #gmsh.model.addPhysicalGroup(3, [ov3[1][1]], 107)
    #gmsh.model.addPhysicalGroup(3, [ov4[1][1]], 108)
    
    gmsh.model.occ.synchronize()
    
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(3)

    gmsh.write("./3dcorpo.msh")
    gmsh.finalize()

    # Read in mesh
    msh = meshio.read("./3dcorpo.msh")
    
    # Create and save one file for the mesh, and one file for the facets 
    tetra_mesh = create_mesh(msh, "tetra")
    tri_mesh = create_mesh(msh, "triangle")
    meshio.write("3dcorpo.xdmf", tetra_mesh)
    meshio.write("3dmt.xdmf", tri_mesh)
    
    with XDMFFile(MPI.COMM_WORLD, "3dcorpo.xdmf", "r") as xdmf:
        msh = xdmf.read_mesh(name="Grid")
        ct = xdmf.read_meshtags(msh, name="Grid")
        msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim-1)
    with XDMFFile(MPI.COMM_WORLD, "3dmt.xdmf", "r") as xdmf:
        ft = xdmf.read_meshtags(msh, name="Grid")

Vd = fem.FunctionSpace(msh, ("Lagrange", 1))

#tdim = msh.topology.dim
#fdim = tdim - 1
#msh.topology.create_connectivity(fdim, tdim)

#facets = np.flatnonzero(mesh.compute_boundary_facets(msh.topology))
#dofs = fem.locate_dofs_topological(Vd, fdim, facets)
#u_boundary = fem.Constant(msh, ScalarType(0.0))
#bc = fem.dirichletbc(u_boundary, dofs, Vd)

x  = ufl.SpatialCoordinate(msh)
Ly = 2.4

u_bottom = ScalarType(0.0)
u_top = ScalarType(1.0)

dofsB = fem.locate_dofs_geometrical(Vd, lambda x: np.isclose(x[1], 0))
dofsT = fem.locate_dofs_geometrical(Vd, lambda x: np.isclose(x[1], Ly))
bc = [fem.dirichletbc(ScalarType(0.0), dofsB, Vd), fem.dirichletbc(ScalarType(1.0), dofsT, Vd)]

u  = ufl.TrialFunction(Vd)
v  = ufl.TestFunction(Vd)
f  = fem.Constant(msh, ScalarType(0.0))
mu = fem.Constant(msh, ScalarType(1.0))
a  = inner(mu * grad(u), grad(v)) * dx
L  = inner(f, v) * dx

one = fem.Constant(msh, ScalarType(1))
area = fem.form(one * dx)
print(fem.assemble_scalar(area))

opts={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_matsolver_type": "mumps"}
#problem = fem.petsc.LinearProblem(a, L, bcs=bc, petsc_options=opts)
problem = fem.petsc.LinearProblem(a, L, bcs=bc, petsc_options={"ksp_type": "gmres", "ksp_rtol":1e-6, "ksp_atol":1e-10, "ksp_max_it": 1000, "pc_type": "none"})

uh = problem.solve()
uh.name = "u"

with io.XDMFFile(msh.comm, "poisson.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(uh)
