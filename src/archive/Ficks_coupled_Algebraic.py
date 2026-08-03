# -----------------
# Package imports
# -----------------
from pathlib import Path

from mpi4py import MPI
from petsc4py.PETSc import ScalarType  # type: ignore

import numpy as np
import ufl
import csv
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import LinearProblem


# -------------------------------------------------
# Parameters
# -------------------------------------------------
C_hat = 0.02          # initial concentration in Reservoir 1 (mol/m^3)

L0 = 0.01             # domain length in x-direction (m)
H0 = 0.01             # domain height in y-direction (m)
B = 0.005             # out-of-plane thickness (m)

T = 7200     # total simulation time (s) = 1 year + (1 year = 604800*70 s)
dt = 100            # time step (s) = 10 minutes

Df = 10e-8            # free diffusion coefficient (m^2/s)
phi = 0.35            # porosity
tau = 10.0            # tortuosity

Deff = phi * Df / tau # effective diffusivity (m^2/s)

# Reservoir volumes
V1 = 0.00025          # 250 mL = 2.5e-4 m^3
V2 = 0.00025          # 250 mL = 2.5e-4 m^3

# Fixed-point iteration controls
fp_tol = 1.0e-10 #Error tolerance
fp_maxit = 25 # maximum number of iteration per time step

# ODE solver tolerances
ode_rtol = 1.0e-8
ode_atol = 1.0e-12


# -------------------------------------------------
# Fick's flux vector J = -Deff * grad(C) (mol/m^2 s)
# -------------------------------------------------
def J(cf):
    return -Deff * ufl.grad(cf)


# -------------------------------------------------
# Create mesh and function spaces
# -------------------------------------------------
msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (L0, H0)),
    n=(32, 32),
    cell_type=mesh.CellType.triangle,
)

V = fem.functionspace(msh, ("Lagrange", 1))                          # scalar concentration space
VV = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))   # vector flux space


# -------------------------------------------------
# Output folder
# -------------------------------------------------
out_folder = Path("out_ficks_coupled")
out_folder.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------
# Boundary location
# -------------------------------------------------
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

dof1 = fem.locate_dofs_topological(V=V, entity_dim=fdim, entities=facet1)
dof2 = fem.locate_dofs_topological(V=V, entity_dim=fdim, entities=facet2)


# -------------------------------------------------
# Boundary tags for flux integration
# -------------------------------------------------
left_id = 1
right_id = 2

facet_indices = np.hstack([facet1, facet2]).astype(np.int32)
facet_markers = np.hstack([
    np.full(facet1.shape, left_id, dtype=np.int32),
    np.full(facet2.shape, right_id, dtype=np.int32),
])

perm = np.argsort(facet_indices)
mt = mesh.meshtags(msh, fdim, facet_indices[perm], facet_markers[perm])

ds = ufl.Measure("ds", domain=msh, subdomain_data=mt)
n = ufl.FacetNormal(msh)


# -------------------------------------------------
# Storage for time history
# -------------------------------------------------
times = []                  # unit: s

C1_vals = []                # concentration in Reservoir 1 (mol/m^3)
C2_vals = []                # concentration in Reservoir 2 (mol/m^3)

J1_vals = []                # total flux from Reservoir 1 into the domain (mol/s)
J_left_2D_vals = []         # 2D line-integrated flux on the left boundary (mol/m s)
J_right_2D_vals = []        # 2D line-integrated flux on the right boundary (mol/m s)
J_left_total_vals = []       # total flux through the left boundary (mol/s)
J_right_total_vals = []      # total flux through the right boundary (mol/s)
N_domain_int_vals = []      # total amount of solute in the domain (mol)

balance_vals = []   
residual_vals = []
rel_residual_vals = []


# -------------------------------------------------
# PDE weak form
# -------------------------------------------------
C = ufl.TrialFunction(V)
q = ufl.TestFunction(V)

# Previous concentration field 
C0 = fem.Function(V)
C0.x.array[:] = 0.0
C0.x.scatter_forward()

a = (phi / dt) * C * q * ufl.dx + Deff * ufl.inner(ufl.grad(C), ufl.grad(q)) * ufl.dx
L = (phi / dt) * C0 * q * ufl.dx


# -------------------------------------------------
# Initial values
# Conservation:
#   C1(0)*V1 + ∫Ω C(x,0)dV + C2(0)*V2 = C_hat*V1
# Since C(x,0)=0 and C1(0)=C_hat, C2(0)=0
# -------------------------------------------------
t = 0.0
C1 = C_hat
C2 = 0.0


# -------------------------------------------------
# Flux projection problem (for ParaView output)
# -------------------------------------------------
w = ufl.TestFunction(VV)
uJ = ufl.TrialFunction(VV)

a_Jc = ufl.inner(w, uJ) * ufl.dx
L_Jc = ufl.inner(w, J(C0)) * ufl.dx

problem_Jc = LinearProblem(
    a_Jc,
    L_Jc,
    petsc_options_prefix="Jc_projection",
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
)


# -------------------------------------------------
# Time loop
# -------------------------------------------------
with io.XDMFFile(msh.comm, out_folder / "ficks.xdmf", "w") as fileC, \
     io.XDMFFile(msh.comm, out_folder / "ficks_flux.xdmf", "w") as fileJ:

    fileC.write_mesh(msh)
    fileJ.write_mesh(msh)

    # Initial output
    fileC.write_function(C0, t)
    Jc = problem_Jc.solve()         # unit: mol/m^2 s
    Jc.x.scatter_forward()
    fileJ.write_function(Jc, t)

#------------------------------------------------------------
# Notes on flux calculations:
#   J(cf) = flux vector field
#   J_left_2D = integrated boundary flux per unit thickness
#   J_left_total = total physical flux
#   J1 = total Reservoir 1-left-boundary flux used in the ODE
#------------------------------------------------------------

    while t < T:
        t_old = t
        t_new = t + dt

        # Start fixed-point iteration from previously accepted values
        C1_iter = C1
        C2_iter = C2
        
        k=1
        while k <= fp_maxit:

            # -----------------------------------------
            # Boundary conditions for this iteration
            # -----------------------------------------
            bc1 = fem.dirichletbc(ScalarType(C1_iter), dofs=dof1, V=V)
            bc2 = fem.dirichletbc(ScalarType(C2_iter), dofs=dof2, V=V)
            bcs = [bc1, bc2]

            problem = LinearProblem(
                a,
                L,
                bcs=bcs,
                petsc_options_prefix="ficks",
                petsc_options={
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                    "ksp_error_if_not_converged": True,
                },
            )

            # -----------------------------------------
            # Solve PDE
            # -----------------------------------------
            Cn = problem.solve()
            Cn.x.scatter_forward()

            # -----------------------------------------
            # Flux integrals with DOMAIN outward normal
            # Multiplied by B to convert 2D boundary integral
            # into total flux through the physical surface
            # -----------------------------------------
            # 2D line-integrated fluxes (per unit out-of-plane thickness)
            J_left_local = fem.assemble_scalar(
                fem.form(ufl.dot(J(Cn), n) * ds(left_id))
            )
            J_right_local = fem.assemble_scalar(
                fem.form(ufl.dot(J(Cn), n) * ds(right_id))
            )
            
            J_left_2D = msh.comm.allreduce(J_left_local, op=MPI.SUM)    # unit: mol/ (m s)
            J_right_2D = msh.comm.allreduce(J_right_local, op=MPI.SUM)  # unit: mol/ (m s)

            # Physical total fluxes through the actual 3D surfaces
            J_left_total = B * J_left_2D ## mol/s
            J_right_total = B * J_right_2D
         

            # -----------------------------------------
            # Reservoir-1 
            # # Since J_left_total is obtained from the domain-outward normal,
            # it is negative when solute leaves reservoir 1 and enters the domain.
            # Therefore using J1 = J_left_total makes dC1/dt negative during
            # reservoir depletion, which is physically correct.
            #
            # Therefore using J1 = J_left_domain makes
            # dC1/dt negative during reservoir depletion,
            # which is physically correct.
            # -----------------------------------------
            J1 = J_left_total

            # -----------------------------------------
            # ODE:
            #   V1 * dC1/dt = J1 =>  dC1/dt = J1/V1 
            # -----------------------------------------
            def rhs_C1(_t, y):
                return np.array([J1 / V1], dtype=np.float64)

            sol = solve_ivp(
                rhs_C1,
                (t_old, t_new),
                y0=np.array([C1], dtype=np.float64),
                method="RK23",
                t_eval=[t_new],
                max_step=dt/2,
                rtol=ode_rtol,
                atol=ode_atol,
            )

            if not sol.success:
                raise RuntimeError(
                    f"ODE solve failed at t = {t_new:.6e}: {sol.message}"
                )

            C1_new = float(sol.y[0, -1])

            # -----------------------------------------
            # Total amount in the domain
            # Multiply by B to convert 
            # into physical volume integral
            # -----------------------------------------
            N_domain_local = fem.assemble_scalar(fem.form(Cn * ufl.dx))
            N_domain_int = B * msh.comm.allreduce(N_domain_local, op=MPI.SUM)   # unit: mol

            # -----------------------------------------
            # Algebraic equation:
            #   C1*V1 + ∫Ω C dV + C2*V2 = C_hat*V1
            #
            # so:
            #   C2 = (C_hat*V1 - C1*V1 - ∫Ω C dV) / V2
            # -----------------------------------------
            C2_new = (C_hat * V1 - C1_new * V1 - N_domain_int) / V2

            # -----------------------------------------
            # Fixed-point convergence check
            # -----------------------------------------
            err = max(abs(C1_new - C1_iter), abs(C2_new - C2_iter))

            if msh.comm.rank == 0:
                print(
                    f"t = {t_new:8.2f} s, iter = {k:2d}, "
                   # f"C1 = {C1_new:.6e}, C2 = {C2_new:.6e}, "
                   # f"J1 = {J1:.6e}, "
                    #f"J_left_2D = {J_left_2D:.6e}, "
                   # f"J_right_2D = {J_right_2D:.6e}, "
                   # f"J_left_total = {J_left_total:.6e}, "
                   # f"J_right_total = {J_right_total:.6e}, "
                  #  f"IntC = {N_domain_int:.6e}, "
                  #  f"err = {err:.3e}"
                )

            # Update iteration guesses
            C1_iter = C1_new
            C2_iter = C2_new

            if err < fp_tol:
                if msh.comm.rank == 0:
                    print(f"Fixed-point iteration converged after {k} iterations.")
                break
            k = k + 1

        else:
            raise RuntimeError(
                f"Fixed-point iteration did not converge at t = {t_new:.6e} "
                f"after {fp_maxit} iterations."
            )

        # -----------------------------------------
        # Accept converged step
        # -----------------------------------------
        t = t_new
        C1 = C1_iter
        C2 = C2_iter

        # -----------------------------------------
        # Conservation check using accepted values
        # -----------------------------------------
        balance = C1 * V1 + N_domain_int + C2 * V2
        target = C_hat * V1
        residual = balance - target
        rel_residual = residual / target if abs(target) > 0 else np.nan

        # Store history
        times.append(t)

        C1_vals.append(C1)
        C2_vals.append(C2)

        J1_vals.append(J1)
        J_left_2D_vals.append(J_left_2D)
        J_right_2D_vals.append(J_right_2D)
        J_left_total_vals.append(J_left_total)
        J_right_total_vals.append(J_right_total)
        N_domain_int_vals.append(N_domain_int)

        balance_vals.append(balance)
        residual_vals.append(residual)
        rel_residual_vals.append(rel_residual)

        # Write accepted concentration field
        fileC.write_function(Cn, t)

        # Update previous concentration for next time step
        C0.x.array[:] = Cn.x.array[:]
        C0.x.scatter_forward()

        # Project and write flux field
        Jc = problem_Jc.solve()
        Jc.x.scatter_forward()
        fileJ.write_function(Jc, t)

        #if msh.comm.rank == 0:
        #    dC1_step = dt * J1 / V1
        #    print(
        ##        f"ACCEPTED: t = {t:8.2f} s, "
        #        f"C1 = {C1:.6e}, C2 = {C2:.6e}, "
        #        f"J1 = {J1:.6e}, IntC = {N_domain_int:.6e}"
        #    )
        #    print(
        #        f"mass check: balance = {balance:.12e}, "
        #        f"target = {target:.12e}, "
         #       f"residual = {residual:.12e}, "
        #        f"relative residual = {rel_residual:.12e}"
        #    )
         #   print(
        #        f"sanity: predicted dC1 over one step = {dC1_step:.6e}"
         #   )


# -------------------------------------------------
# Save history to CSV
# -------------------------------------------------
if msh.comm.rank == 0:
    csv_path = out_folder / "history.csv"

    with open(csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "time_s",
            "C1(mol/m^3)",
            "C2(mol/m^3)",
            "J1(mol/s)",
            "J_left_2D(mol/(m s))",
            "J_right_2D(mol/(m s))",
            "J_left_total(mol/s)",
            "J_right_total(mol/s)",
            "Int_C_domain(mol)",
            "balance(mol)",
            "target(mol)",
            "residual(mol)",
            "relative_residual"
        ])

        for ti, c1i, c2i, j1i, jldi, jrdi, jlti, jrti, cinti, bali, resi, rresi in zip(
            times,
            C1_vals,
            C2_vals,
            J1_vals,
            J_left_2D_vals,
            J_right_2D_vals,
            J_left_total_vals,
            J_right_total_vals,
            N_domain_int_vals,
            balance_vals,
            residual_vals,
            rel_residual_vals
        ):
            writer.writerow([
                ti,
                c1i,
                c2i,
                j1i,
                jldi,
                jrdi,
                jlti,
                jrti,
                cinti,
                bali,
                C_hat * V1,
                resi,
                rresi
            ])

    print(f"History saved to: {csv_path}")

# -------------------------------------------------
# Plot results using matplotlib
# -------------------------------------------------
if msh.comm.rank == 0:
    # ---------------------------------------------
    # Convert lists to numpy arrays
    # ---------------------------------------------
    time_arr = np.array(times, dtype=float)              # s
    time_days = time_arr / 3600.0                          # days

    Ndot_left_arr = np.array(J_left_total_vals, dtype=float)    # mol/s
    Ndot_right_arr = np.array(J_right_total_vals, dtype=float)  # mol/s
    N_domain_arr = np.array(N_domain_int_vals, dtype=float)     # mol
    
      
    # ---------------------------------------------
    # Figure style settings
    # ---------------------------------------------
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "lines.linewidth": 2.0,
    })

    # ---------------------------------------------
    # 1) Left and right boundary molar transport rates
    # ---------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(time_days, Ndot_left_arr, label="Left boundary molar transport rate")
    ax1.plot(time_days, Ndot_right_arr, label="Right boundary molar transport rate")
    ax1.axhline(0.0, linestyle="--", linewidth=1.0)
    ax1.set_xlabel("Time (hrs)")
    ax1.set_ylabel("Molar transport rate (mol/s)")
    ax1.set_title("Boundary Molar Transport Rates vs Time")
    ax1.grid(True)
    ax1.legend()
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig1.tight_layout()
    fig1.savefig(out_folder / "boundary_molar_transport_rates_vs_time.png")
    
    # ---------------------------------------------
    # 2) Total solute amount in domain
    # ---------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(time_days, N_domain_arr)
    ax2.set_xlabel("Time (hrs)")
    ax2.set_ylabel("Total solute amount in domain (mol)")
    ax2.set_title("Total Solute Amount in Domain vs Time")
    ax2.grid(True)
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig2.tight_layout()
    fig2.savefig(out_folder / "total_solute_amount_in_domain_vs_time.png")
    