from mpi4py import MPI

try:
    import gmsh
except ModuleNotFoundError:
    import sys
    print("This demo requires gmsh to be installed")
    sys.exit(0)


def generate_mesh(filename: str, radius: int, length: int, h_elem: int, order: int, verbose: bool = False):
    if MPI.COMM_WORLD.rank == 0:

        gmsh.initialize()
        gmsh.model.add("helmholtz_domain")
        gmsh.option.setNumber("General.Terminal", verbose)
        # Set the mesh size
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.75*h_elem)

        # Add scatterers
        c1 = gmsh.model.occ.addCircle(0.0, 0.0, 0.0, radius)
        gmsh.model.occ.addCurveLoop([c1], tag=c1)
        gmsh.model.occ.addPlaneSurface([c1], tag=c1)

        # Add domain
        r0 = gmsh.model.occ.addRectangle(-length/2, -length/2, 0.0, length, length)
        inclusive_rectangle, _ = gmsh.model.occ.fragment([(2, r0)], [(2, c1)])

        gmsh.model.occ.synchronize()

        # Add physical groups
        gmsh.model.addPhysicalGroup(2, [c1], tag=1)
        gmsh.model.addPhysicalGroup(2, [r0], tag=2)

        # Generate mesh
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(order)
        gmsh.model.mesh.optimize("HighOrder")
        gmsh.write(filename)

        gmsh.finalize()
