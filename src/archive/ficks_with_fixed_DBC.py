## Preliminaries
from pathlib import Path

from mpi4py import MPI
from petsc4py.PETSc import ScalarType  # type: ignore

import numpy as np
import ufl

from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import LinearProblem


L0 = 0.01
H0 = 0.01
# Create mesh and function space
msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (L0, H0)),
    n=(32, 32),
    cell_type=mesh.CellType.triangle,
)
V = fem.functionspace(msh, ("Lagrange", 1))


## Output folder
out_folder = Path("out_ficks")
out_folder.mkdir(parents=True, exist_ok=True)


## Dirichlet boundary conditions
fdim = msh.topology.dim - 1

facet1 = mesh.locate_entities_boundary(
    msh,
    dim=fdim,
    marker=lambda x: np.isclose(x[0], 0.0) 
)
facet2 = mesh.locate_entities_boundary(
    msh,
    dim=fdim,
    marker=lambda x: np.isclose(x[0], L0)
)
# the degrees-of-freedom 
dof1 = fem.locate_dofs_topological(V=V, entity_dim=fdim, entities=facet1)
dof2 = fem.locate_dofs_topological(V=V, entity_dim=fdim, entities=facet2)
# represents the boundary condition:
bc1 = fem.dirichletbc(value=ScalarType(0.02), dofs=dof1, V=V)
bc2 = fem.dirichletbc(value=ScalarType(0.0), dofs=dof2, V=V)

bcs=[bc1, bc2]


# Problem definition:
T = 21600
dt = 864
Df = 10e-8
phi = 0.35
tau = 10

C = ufl.TrialFunction(V)
q = ufl.TestFunction(V)

C0 = fem.Function(V)
C0.x.array[:] = 0.0
C0.x.scatter_forward()

a = (phi/dt) * C * q * ufl.dx + (phi*Df / tau) * ufl.inner(ufl.grad(C), ufl.grad(q)) * ufl.dx
L = (phi/dt) * C0 * q * ufl.dx
 
problem = LinearProblem(
    a,
    L,
    bcs=bcs,
    petsc_options_prefix="ficks",
    petsc_options={"ksp_type": "preonly", "pc_type": "lu", "ksp_error_if_not_converged": True},
)

t = 0.0
with io.XDMFFile(msh.comm, out_folder / "ficks.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(C0, t)

    while t < T:
        t += dt

        Cn = problem.solve()
        
        file.write_function(Cn, t)

        C0.x.array[:] = Cn.x.array[:]
        C0.x.scatter_forward()