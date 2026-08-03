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

from dolfinx import fem, io, mesh, geometry
from dolfinx.fem.petsc import LinearProblem


# -------------------------------------------------
# Parameters
# -------------------------------------------------
C_hat = 0.02          # initial concentration in Reservoir 1 (mol/m^3)

L0 = 0.01             # domain length in x-direction (m)
H0 = 0.01             # domain height in y-direction (m)
B = 0.005             # out-of-plane thickness (m)

T = 36000             # total simulation time (s) = 10 hours
dt = 30               # time step (s)

Df = 10e-8            # free diffusion coefficient (m^2/s)
phi = 0.35            # porosity
tau = 10.0            # tortuosity

Deff = phi * Df / tau # effective diffusivity (m^2/s)

# Reservoir volumes
V1 = 0.00025          # 250 mL = 2.5e-4 m^3
V2 = 0.00025          # 250 mL = 2.5e-4 m^3

# Fixed-point iteration controls
fp_tol = 1.0e-10
fp_maxit = 25

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

V = fem.functionspace(msh, ("Lagrange", 1))                         # scalar concentration space
VV = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))  # vector flux space


# -------------------------------------------------
# Output folder
# -------------------------------------------------
out_folder = Path("out_ficks_coupled_2_ODEs_Final")
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
times = []                  # s

C1_vals = []                # mol/m^3
C2_vals = []                # mol/m^3

J1_vals = []                # mol/s
J2_vals = []                # mol/s
J_left_2D_vals = []         # mol/(m s)
J_right_2D_vals = []        # mol/(m s)
J_left_total_vals = []      # mol/s
J_right_total_vals = []     # mol/s
N_domain_int_vals = []      # mol

balance_vals = []
residual_vals = []
rel_residual_vals = []


# -------------------------------------------------
# Selected concentration profiles C(x,t)
# -------------------------------------------------
selected_steps_requested = [1, 10, 20, 80, 300]

nsteps_total = int(round(T / dt))
selected_steps = [s for s in selected_steps_requested if s <= nsteps_total]

if msh.comm.rank == 0 and len(selected_steps) < len(selected_steps_requested):
    print(
        f"Warning: requested steps {selected_steps_requested}, "
        f"but this run only has {nsteps_total} time steps. "
        f"Using {selected_steps}."
    )

x_profile = np.linspace(0.0, L0, 300)   # x locations for the profile
y_profile = 0.5 * H0                    # mid-height line

profile_results = {}                    # step -> {"time_s": ..., "C": ...}
step_counter = 0


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
# -------------------------------------------------
t = 0.0
C1 = C_hat
C2 = 0.0

# Store initial state at t = 0
times.append(0.0)

C1_vals.append(C1)
C2_vals.append(C2)

J1_vals.append(0.0)
J2_vals.append(0.0)
J_left_2D_vals.append(0.0)
J_right_2D_vals.append(0.0)
J_left_total_vals.append(0.0)
J_right_total_vals.append(0.0)
N_domain_int_vals.append(0.0)

balance0 = C1 * V1 + 0.0 + C2 * V2
target0 = C_hat * V1
residual0 = balance0 - target0
rel_residual0 = residual0 / target0 if abs(target0) > 0 else np.nan

balance_vals.append(balance0)
residual_vals.append(residual0)
rel_residual_vals.append(rel_residual0)


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
# Helper: sample scalar FEM function along a line
# -------------------------------------------------
def sample_scalar_along_line(u, x_coords, y_value):
    """
    Sample scalar FEM function u at points (x, y_value).
    Best used in serial runs.
    """
    points = np.zeros((len(x_coords), 3), dtype=np.float64)
    points[:, 0] = x_coords
    points[:, 1] = y_value
    points[:, 2] = 0.0

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points)

    values = np.full(len(x_coords), np.nan, dtype=np.float64)

    points_on_proc = []
    cells = []
    point_ids = []

    for i, p in enumerate(points):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(p)
            cells.append(links[0])
            point_ids.append(i)

    if len(points_on_proc) > 0:
        points_on_proc = np.array(points_on_proc, dtype=np.float64)
        vals = u.eval(points_on_proc, cells)
        vals = np.asarray(vals).reshape(-1)
        values[np.array(point_ids, dtype=np.int32)] = vals

    return values


# -------------------------------------------------
# Time loop
# -------------------------------------------------
with io.XDMFFile(msh.comm, out_folder / "ficks.xdmf", "w") as fileC, \
     io.XDMFFile(msh.comm, out_folder / "ficks_flux.xdmf", "w") as fileJ:

    fileC.write_mesh(msh)
    fileJ.write_mesh(msh)

    # Initial output
    fileC.write_function(C0, t)
    Jc = problem_Jc.solve()
    Jc.x.scatter_forward()
    fileJ.write_function(Jc, t)

    while t < T:
        t_old = t
        t_new = t + dt

        # Start fixed-point iteration from previously accepted values
        C1_iter = C1
        C2_iter = C2

        k = 1
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
            # Flux integrals with domain outward normal
            # -----------------------------------------
            J_left_local = fem.assemble_scalar(
                fem.form(ufl.dot(J(Cn), n) * ds(left_id))
            )
            J_right_local = fem.assemble_scalar(
                fem.form(ufl.dot(J(Cn), n) * ds(right_id))
            )

            J_left_2D = msh.comm.allreduce(J_left_local, op=MPI.SUM)
            J_right_2D = msh.comm.allreduce(J_right_local, op=MPI.SUM)

            J_left_total = B * J_left_2D
            J_right_total = B * J_right_2D

            # -----------------------------------------
            # Reservoir molar transport rates
            # -----------------------------------------
            J1 = J_left_total
            J2 = J_right_total

            # -----------------------------------------
            # Coupled reservoir ODEs
            # -----------------------------------------
            def reservoirs(_t, y):
                return np.array([
                    J1 / V1,
                    J2 / V2
                ], dtype=np.float64)

            sol = solve_ivp(
                reservoirs,
                (t_old, t_new),
                y0=np.array([C1, C2], dtype=np.float64),
                method="RK23",
                t_eval=[t_new],
                max_step=dt / 2,
                rtol=ode_rtol,
                atol=ode_atol,
            )

            if not sol.success:
                raise RuntimeError(
                    f"Reservoir ODE solve failed at t = {t_new:.6e}: {sol.message}"
                )

            C1_new = float(sol.y[0, -1])
            C2_new = float(sol.y[1, -1])

            # -----------------------------------------
            # Total amount in the domain
            # -----------------------------------------
            N_domain_local = fem.assemble_scalar(fem.form(Cn * ufl.dx))
            N_domain_int = B * msh.comm.allreduce(N_domain_local, op=MPI.SUM)

            # -----------------------------------------
            # Fixed-point convergence check
            # -----------------------------------------
            err = max(abs(C1_new - C1_iter), abs(C2_new - C2_iter))

            if msh.comm.rank == 0:
                print(f"t = {t_new:8.2f} s, iter = {k:2d}")

            C1_iter = C1_new
            C2_iter = C2_new

            if err < fp_tol:
                if msh.comm.rank == 0:
                    print(f"Fixed-point iteration converged after {k} iterations.")
                break

            k += 1

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
        step_counter += 1

        # -----------------------------------------
        # Conservation check
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
        J2_vals.append(J2)
        J_left_2D_vals.append(J_left_2D)
        J_right_2D_vals.append(J_right_2D)
        J_left_total_vals.append(J_left_total)
        J_right_total_vals.append(J_right_total)
        N_domain_int_vals.append(N_domain_int)

        balance_vals.append(balance)
        residual_vals.append(residual)
        rel_residual_vals.append(rel_residual)

        # -----------------------------------------
        # Store selected concentration profiles C(x,t)
        # -----------------------------------------
        if step_counter in selected_steps:
            c_profile = sample_scalar_along_line(Cn, x_profile, y_profile)

            if msh.comm.rank == 0:
                profile_results[step_counter] = {
                    "time_s": t,
                    "C": c_profile.copy()
                }
                print(
                    f"Stored concentration profile at step {step_counter}, "
                    f"t = {t:.2f} s"
                )

        # Write accepted concentration field
        fileC.write_function(Cn, t)

        # Update previous concentration for next time step
        C0.x.array[:] = Cn.x.array[:]
        C0.x.scatter_forward()

        # Project and write flux field
        Jc = problem_Jc.solve()
        Jc.x.scatter_forward()
        fileJ.write_function(Jc, t)


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
            "J2(mol/s)",
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

        for ti, c1i, c2i, j1i, j2i, jldi, jrdi, jlti, jrti, cinti, bali, resi, rresi in zip(
            times,
            C1_vals,
            C2_vals,
            J1_vals,
            J2_vals,
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
                j2i,
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
# Save selected concentration profiles to CSV
# -------------------------------------------------
if msh.comm.rank == 0 and len(profile_results) > 0:
    profile_csv_path = out_folder / "selected_concentration_profiles.csv"

    sorted_steps = sorted(profile_results.keys())

    with open(profile_csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        header = ["x_m", "x_mm"]
        for s in sorted_steps:
            t_s = profile_results[s]["time_s"]
            header.append(f"step_{s}_t_{t_s:.0f}_s")
        writer.writerow(header)

        for i, x in enumerate(x_profile):
            row = [x, x * 1000.0]
            for s in sorted_steps:
                row.append(profile_results[s]["C"][i])
            writer.writerow(row)

    print(f"Selected concentration profiles saved to: {profile_csv_path}")


# -------------------------------------------------
# Plot results using matplotlib
# -------------------------------------------------
if msh.comm.rank == 0:
    time_arr = np.array(times, dtype=float)
    time_hrs = time_arr / 3600.0

    Ndot_left_arr = np.array(J_left_total_vals, dtype=float)
    Ndot_right_arr = np.array(J_right_total_vals, dtype=float)
    N_domain_arr = np.array(N_domain_int_vals, dtype=float)

    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 800,
        "font.size": 28,
        "axes.titlesize": 28,
        "axes.labelsize": 24,
        "legend.fontsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "lines.linewidth": 2.8,
        "axes.linewidth": 1.2,
    })

    # ---------------------------------------------
    # 1) Boundary molar transport rates vs time
    # ---------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    ax1.plot(
        time_hrs, Ndot_left_arr,
        label=r"$\dot{Q}_\mathrm{left}$ (Reservoir 1 $\rightarrow$ domain)",
        color="tab:blue", linestyle="-"
    )
    ax1.plot(
        time_hrs, Ndot_right_arr,
        label=r"$\dot{Q}_\mathrm{right}$ (domain $\rightarrow$ Reservoir 2)",
        color="tab:orange", linestyle="-"
    )

    ax1.axhline(0.0, linestyle=":", linewidth=1.2, color="gray")
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Molar transport rate (mol/s)")
    ax1.set_title("Boundary Molar Transport Rates vs Time", fontweight="bold", pad=16)

    ax1.grid(False)
    ax1.minorticks_on()
    ax1.tick_params(which="both", direction="in", color="gray", length=6, width=1.1)
    ax1.tick_params(which="minor", length=3.5)
    ax1.legend(loc="center right", frameon=True)
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig1.tight_layout()
    fig1.savefig(out_folder / "boundary_molar_transport_rates_vs_time.png", bbox_inches="tight")
    fig1.savefig(out_folder / "boundary_molar_transport_rates_vs_time.pdf", bbox_inches="tight")

    # ---------------------------------------------
    # 2) Total solute amount in domain
    # ---------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    ax2.plot(time_hrs, N_domain_arr, color="tab:green", linestyle="-")
    ax2.set_xlabel("Time (hours)")
    ax2.set_ylabel("Total amount of solute in domain (mol)")
    ax2.set_title("Total Solute Amount in Domain vs Time (First 10 hours)", fontweight="bold", pad=16)

    ax2.grid(False)
    ax2.minorticks_on()
    ax2.tick_params(which="both", direction="in", color="gray", length=6, width=1.1)
    ax2.tick_params(which="minor", length=3.5)
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig2.tight_layout()
    fig2.savefig(out_folder / "total_solute_amount_in_domain_vs_time.png", bbox_inches="tight")
    fig2.savefig(out_folder / "total_solute_amount_in_domain_vs_time.pdf", bbox_inches="tight")

    # ---------------------------------------------
    # 3) Selected concentration profiles C(x,t)
    # ---------------------------------------------
    if len(profile_results) > 0:
        fig3, ax3 = plt.subplots(figsize=(13, 8.5))

    # Convert x-coordinate from m to mm
    x_mm = x_profile * 1000.0

    # Plot in ascending time-step order
    sorted_steps = sorted(profile_results.keys())

    # Optional fixed colors for consistency
    profile_colors = {
        1:   "tab:blue",
        10:  "tab:orange",
        20:  "tab:green",
        80:  "tab:red",
        300: "tab:purple",
    }

    # Store last curve for annotation
    last_x = None
    last_y = None

    for s in sorted_steps:
        c_prof = profile_results[s]["C"]

        # Custom legend labels
        if s == 1:
            label = r"$t = 30\,\mathrm{s}$"
        elif s == 10:
            label = r"$t = 5\,\mathrm{min}$"
        elif s == 20:
            label = r"$t = 10\,\mathrm{min}$"
        elif s == 80:
            label = r"$t = 40\,\mathrm{min}$"
        elif s == 300:
            label = r"$t = 2.5\,\mathrm{h}$"
        else:
            t_s = profile_results[s]["time_s"]
            if t_s < 60.0:
                label = fr"$t = {t_s:.0f}\,\mathrm{{s}}$"
            elif t_s < 3600.0:
                label = fr"$t = {t_s/60.0:.1f}\,\mathrm{{min}}$"
            else:
                label = fr"$t = {t_s/3600.0:.2f}\,\mathrm{{h}}$"

        ax3.plot(
            x_mm,
            c_prof,
            label=label,
            color=profile_colors.get(s, None),
            linewidth=2.8
        )

        if s == sorted_steps[-1]:
            last_x = x_mm.copy()
            last_y = c_prof.copy()

    # Axis labels and title
    ax3.set_xlabel("Distance, $x$ (mm)", fontsize=24)
    ax3.set_ylabel(r"Concentration, $C$ (mol/m$^3$)", fontsize=24)
    ax3.set_title("Concentration Profiles Through the Domain",
                  fontsize=28, fontweight="bold", pad=16)

    # Focus the y-range near the actual concentration range
    ax3.set_ylim(0.0, 0.0205)
    ax3.set_xlim(0.0, 10.0)

    # Cleaner ticks for poster quality
    ax3.grid(False)
    ax3.minorticks_on()
    ax3.tick_params(which="both", direction="in", length=6, width=1.1, labelsize=18)
    ax3.tick_params(which="minor", length=3.5)

    # Make y-axis show normal decimal values instead of scientific notation
    ax3.ticklabel_format(axis="y", style="plain")

    # Legend
    ax3.legend(
        loc="upper right",
        frameon=True,
        fontsize=18
    )

    # Annotate the last profile as near steady state
    if last_x is not None and last_y is not None:
        # choose a point around x = 6 mm for annotation target
        idx_annot = np.argmin(np.abs(last_x - 6.0))
        ax3.annotate(
            "near steady state",
            xy=(last_x[idx_annot], last_y[idx_annot]),
            xytext=(6, 0.01),
            textcoords="data",
            fontsize=18,
            color="black",
            arrowprops=dict(
                arrowstyle="->",
                linewidth=1.6,
                color="black"
            )
        )

    fig3.tight_layout()
    fig3.savefig(out_folder / "selected_concentration_profiles.png", dpi=800, bbox_inches="tight")