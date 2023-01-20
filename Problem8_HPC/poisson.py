from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from dolfinx.fem.petsc import LinearProblem
import ufl
import numpy


def create_mesh(comm, target_dofs, strong_scaling):
    # Get number of processes
    num_processes = comm.size

    # Target total dofs
    N = target_dofs if strong_scaling else target_dofs * num_processes

    # Compute the number of cells in each direction
    Nc = numpy.array([1, 1, 1], dtype=int)
    while numpy.prod(Nc+1) < N:
        for i in range(len(Nc)):
            Nc[i] = Nc[i]+1
            if numpy.prod(Nc+1) >= N:
                break

    mesh = dolfinx.mesh.create_unit_cube(comm, Nc[0], Nc[1], Nc[2])
    return mesh


def poisson_solver(ndofs: int, petsc_options: dict):
    comm = MPI.COMM_WORLD
    mesh = create_mesh(comm, ndofs, False)
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)

    V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    bndry_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    bndry_dofs = dolfinx.fem.locate_dofs_topological(V, tdim - 1, bndry_facets)
    zero = dolfinx.fem.Constant(mesh, 0.0)
    bcs = [dolfinx.fem.dirichletbc(zero, bndry_dofs, V)]

    x = ufl.SpatialCoordinate(mesh)
    f = 1 - x[0]**2

    a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
    L = f*v*ufl.dx

    uh = dolfinx.fem.Function(V)
    problem = LinearProblem(a, L, u=uh, bcs=bcs, petsc_options=petsc_options)

    with dolfinx.common.Timer("Solver") as t:
        problem.solve()

    num_dofs = V.dofmap.index_map.size_global
    if comm.rank == 0:
        print(f"Number of degrees of freedom: {num_dofs}", flush=True)
        print(f"Solver time: {t.elapsed()[0]}", flush=True)
        print(f"Number of iterations {problem.solver.its}", flush=True)

    return num_dofs, t.elapsed()[0], problem.solver.its


# Solver configuration
lu_solver = {"ksp_type": "preonly", "pc_type": "lu"}
cg_nopre = {"ksp_type": "cg", "pc_type": "none"}
multigrid = {"ksp_type": "cg", "pc_type": "hypre",
             "pc_hypre_type": "boomeramg",
             "ksp_rtol": "1e-7"}

if __name__ == "__main__":
    dofs = 200000
    poisson_solver(dofs, cg_nopre)
