from mpi4py import MPI

try:
    import gmsh
except ModuleNotFoundError:
    import sys
    print("This demo requires gmsh to be installed")
    sys.exit(0)


def generate_mesh(filename: str, lmbda: int , order: int, verbose:bool=False):
    if MPI.COMM_WORLD.rank == 0:

        gmsh.initialize()
        gmsh.model.add("helmholtz_domain")
        gmsh.option.setNumber("General.Terminal", verbose)
        # Set the mesh size
        gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1.5*lmbda)


        # Add scatterers
        c1 = gmsh.model.occ.addCircle(0.0, -1.1*lmbda, 0.0, 0.8*lmbda)
        gmsh.model.occ.addCurveLoop([c1], tag=c1)
        gmsh.model.occ.addPlaneSurface([c1], tag=c1)

        c2 = gmsh.model.occ.addCircle(0.0, 1.1*lmbda, 0.0, 0.8*lmbda)
        gmsh.model.occ.addCurveLoop([c2], tag=c2)
        gmsh.model.occ.addPlaneSurface([c2], tag=c2)

        # Add domain
        r0 = gmsh.model.occ.addRectangle(
            -5*lmbda, -5*lmbda, 0.0, 10*lmbda, 10*lmbda)
        inclusive_rectangle, _ = gmsh.model.occ.fragment(
            [(2, r0)], [(2, c1), (2, c2)])

        gmsh.model.occ.synchronize()

        # Add physical groups
        gmsh.model.addPhysicalGroup(2, [c1, c2], tag=1)
        gmsh.model.addPhysicalGroup(2, [r0], tag=2)

        # Generate mesh
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(order)
        gmsh.model.mesh.optimize("HighOrder")
        gmsh.write(filename)
        
        gmsh.finalize()