from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from dolfinx.fem.petsc import LinearProblem
import ufl
import numpy
import argparse


def create_mesh(comm, target_dofs, strong_scaling):
    """Create a mesh such that the target number of dofs is achieved.

    Parameters
    ----------
    comm : object
        Communication object used to specify the number of processes in the simulation.
    target_dofs : int
        Target number of degrees of freedom (dofs) for the mesh.
        The number of dofs represents the number of unknown variables in the simulation.
    strong_scaling : bool
        Boolean flag indicating whether the simulation is using strong scaling.
        Strong scaling means that the number of dofs is kept constant while the number of processes is increased.

    Returns
    -------
    mesh : object
        The created mesh using the `create_unit_cube` function with the specified number of cells
        in each direction and the communication object.

    """
    # Get number of processes
    num_processes = comm.size

    # If strong_scaling is true, the target total dofs is simply equal to target_dofs.
    # If strong_scaling is false, the target total dofs is equal to target_dofs
    # multiplied by the number of processes.
    N = target_dofs if strong_scaling else target_dofs * num_processes

    # Computes the number of cells in each direction of the mesh by starting with
    # Nc = [1, 1, 1] and incrementing each dimension until the total number of cells
    # is greater than or equal to the target total dofs.
    Nc = numpy.array([1, 1, 1], dtype=int)
    while numpy.prod(Nc+1) < N:
        for i in range(len(Nc)):
            Nc[i] = Nc[i]+1
            if numpy.prod(Nc+1) >= N:
                break

    return dolfinx.mesh.create_unit_cube(comm, Nc[0], Nc[1], Nc[2])


def poisson_solver(ndofs: int, petsc_options: dict, strong_scaling: bool = False):
    """
    Solves the Poisson equation using a finite element method.

    Parameters
    ----------
    ndofs : int
        The target number of degrees of freedom.
    petsc_options : dict
        Dictionary of PETSc solver options.
    strong_scaling : bool, optional
        Whether to enable strong scaling or not, by default False.

    Returns
    -------
    num_dofs : int
        The number of degrees of freedom.
    solver_time : float
        The time taken by the solver to complete.
    num_iterations : int
        The number of iterations taken by the solver.

    """
    comm = MPI.COMM_WORLD
    mesh = create_mesh(comm, ndofs, strong_scaling)
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
        print(f"Number of iterations: {problem.solver.its}", flush=True)
        print(f"Number of MPI processes: {comm.size}", flush=True)
        print(f"Solver: {petsc_options}", flush=True)

    return num_dofs, t.elapsed()[0], problem.solver.its


# Solver configuration
lu_solver = {"ksp_type": "preonly",
             "pc_type": "lu"}

cg_nopre = {"ksp_type": "cg",
            "pc_type": "none",
            "ksp_rtol": 1e-7}

multigrid = {"ksp_type": "cg",
             "pc_type": "hypre",
             "pc_hypre_type": "boomeramg",
             "pc_hypre_boomeramg_strong_threshold": 0.7,
             "pc_hypre_boomeramg_agg_nl": 4,
             "pc_hypre_boomeramg_agg_num_paths": 2,
             "ksp_rtol": 1e-7}

solvers = {"lu": lu_solver, "cg": cg_nopre, "mg": multigrid}
scaling = {"strong": True, "weak": False}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Code to test the parallel performance of DOLFINx and the underlying linear solvers.')
    parser.add_argument('--num_dofs',  type=int,
                        help='Number of degrees-of-freedom: total (in case of strong scaling) or per process (for weak scaling).')
    parser.add_argument('--solver', type=str, default="cg",
                        help='Solver to use', choices=['lu', 'cg', 'mg'])
    parser.add_argument('--scaling_type', type=str, default="strong",
                        help='Scaling type: strong (fixed problem size) or weak (fixed problem size per process)', choices=['strong', 'weak'])

    args = parser.parse_args()
    num_dofs = args.num_dofs
    solver = solvers[args.solver]
    strong_scaling = scaling[args.scaling_type]
    poisson_solver(num_dofs, solver, strong_scaling)
