#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 11:27:25 2022

@author: luciano
"""

import gmsh
import numpy as np
import matplotlib.pyplot as plt


from dolfinx import fem, io, mesh, plot
from dolfinx.fem import (Constant, dirichletbc, Function, FunctionSpace, assemble_scalar,
                         form, locate_dofs_geometrical, locate_dofs_topological, Constant)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_rectangle, locate_entities, CellType, locate_entities_boundary
from dolfinx.plot import create_vtk_mesh

from ufl import (SpatialCoordinate, TestFunction, TrialFunction, Measure,
                 dx, grad, div, inner, dot, as_vector, Circumradius, sqrt, exp,
                 conditional,lt,tr, Identity,FacetNormal,nabla_div,sym)

from mpi4py import MPI
from petsc4py.PETSc import ScalarType


Lx = 0.4
Ly = 2.4

bottom_wall = 0
top_wall    = 1

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

    lc = 0.065

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
 

V = fem.VectorFunctionSpace(msh, ("CG", 1))
u  = TrialFunction(V)
v  = TestFunction(V)

fdim = msh.topology.dim - 1

def bottom(x):
    return np.isclose(x[1], 0)

def top(x):
    return np.isclose(x[1], Ly)

#bfacets = locate_entities_boundary(msh, fdim, bottom)
#tfacets = locate_entities_boundary(msh, fdim, top)

#u_bottom = np.array([0,0,0], dtype=ScalarType)
#u_top = np.array([0,0.1,0], dtype=ScalarType)
u_bottom = as_vector([0, 0, 0])
u_top = as_vector([0,0.1,0])

#bcs = []
#bcs.append(dirichletbc(u_bottom, locate_dofs_topological(V, fdim, bfacets), V))
#bcs.append(dirichletbc(u_top, locate_dofs_topological(V, fdim, tfacets), V))

dofsB = locate_dofs_geometrical(V, lambda x: np.isclose(x[1], 0))
dofsT = locate_dofs_geometrical(V, lambda x: np.isclose(x[1], Ly))
bcs = [dirichletbc(u_botton, dofsB, V), dirichletbc(u_top, dofsT, V)]

x = SpatialCoordinate(msh)

f = Constant(msh, ScalarType((0,0,0)))
F = Constant(msh, ScalarType((0,0,0)))

E = 10.0

nu = 0.3

mu = E/(2.0*(1.0 + nu))

lmbda = E*nu/((1.0 + nu)*(1.0 - 2.0*nu))

def epsilon(u):
    return sym(grad(u)) # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
def sigma(u):
    return lmbda * nabla_div(u) * Identity(u.geometric_dimension()) + 2*mu*epsilon(u)

a = inner(sigma(u), epsilon(v)) * dx
#a = 2*np.pi*(2*mu*(inner(sigma(u),sigma(v))) + u.sub(0)*v.sub(0)/x[0]*x[0])*x[0]*dx + 2*np.pi*(lmbda*(div(u)+u.sub(0)/x[0])*(div(v)+v.sub(0)/x[0]))*x[0]*dx
#a = 2*np.pi*(2*mu*(inner(sigma(u),epsilon(v))) + u[0]*v[0]/x[0]*x[0])*x[0]*dx + 2*np.pi*(lmbda*(div(u)+u[0]/x[0])*(div(v)+v[0]/x[0]))*x[0]*dx

#ds = Measure("ds")( subdomain_data = ft )
L = dot(f, v) *dx #+ dot(F, v) * ds

#problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "gmres", "ksp_rtol":1e-6, "ksp_atol":1e-10, "ksp_max_it": 1000, "pc_type": "none"})
uh = problem.solve()
uh.name = "Deformation"

# Save the results
with XDMFFile(MPI.COMM_WORLD, "deformation.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh)
    xdmf.write_function(uh)

   
s = sigma(uh) -1./3*tr(sigma(uh))*Identity(uh.geometric_dimension())
von_Mises = sqrt(3./2*inner(s, s))

V_von_mises = fem.FunctionSpace(msh, ("DG", 0))
stress_expr = fem.Expression(von_Mises, V_von_mises.element.interpolation_points)
stresses = fem.Function(V_von_mises)
stresses.interpolate(stress_expr)
stresses.name = "Stresses"

with io.XDMFFile(msh.comm, "stresses.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(stresses)   


