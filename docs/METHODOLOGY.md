# Numerical Methodology

## 1. Scope and validation status

This document describes a computational model for H⁺ diffusion through a saturated kaolin-clay domain coupled to two external reservoirs.

**No physical through-diffusion experiment has been carried out for this project.** All concentration fields, transport rates, reservoir histories, and mass inventories discussed below are numerical simulation predictions. The model has not been calibrated or validated against laboratory data.

## 2. Research question

The model addresses the following computational question:

> How do the H⁺ concentration field, boundary molar transport rates, and well-mixed reservoir concentrations evolve when an initially acidic reservoir and an initially solute-free reservoir are connected through a saturated kaolin-clay domain?

The model is intended to support the design and future interpretation of a physical through-diffusion experiment.

## 3. Physical system represented

The simulated system consists of:

- reservoir 1, on the left;
- a saturated kaolin-clay domain, in the center;
- reservoir 2, on the right.

The numerical domain is a two-dimensional rectangle:

$$
\Omega=[0,L_0]\times[0,H_0],
\qquad
L_0=H_0=0.01\ \mathrm{m}.
$$

A constant out-of-plane thickness,

$$
B=0.005\ \mathrm{m}
$$

is used to convert two-dimensional area and line integrals into modeled three-dimensional amounts and rates.

Both reservoirs are assumed to be perfectly mixed and of constant volume:

$$
V_1=V_2=2.5\times10^{-4}\ \mathrm{m^3}.
$$

These values define the computational setup; they are not measurements from a physical experiment.

## 4. Governing equation

The concentration field $C(\mathbf{x},t)$ is governed by

$$
\phi\frac{\partial C}{\partial t}
=
\nabla\cdot\left(D_{\mathrm{eff}}\nabla C\right)
\qquad \text{in }\Omega.
$$

where $\phi$ is the (dimensionless) porosity and $D_{\mathrm{eff}}$ is the effective diffusivity, given by


$$
D_{\mathrm{eff}}
=
\frac{\phi D_f}{\tau},
$$

Here $D_f$ is the free-solution diffusivity of H⁺ and $\tau$ is a dimensionless tortuosity factor. With


$$
D_f=1.0\times10^{-7}\ \mathrm{m^2/s},
\qquad
\phi=0.35,
\qquad
\tau=10.
$$

this gives

$$
D_{\mathrm{eff}}
=
3.5\times10^{-9}\ \mathrm{m^2/s}.
$$

The Fickian flux vector is

$$
\mathbf{J}
=
-D_{\mathrm{eff}}\nabla C,
\qquad
[\mathbf{J}]
=
\mathrm{mol\,m^{-2}\,s^{-1}}.
$$
The values of $D_f$, $\phi$, and $\tau$ above are illustrative parameters chosen during model development

## 5. Boundary conditions

The left and right boundaries are

$$
\Gamma_1=\{x=0\},
\qquad
\Gamma_2=\{x=L_0\}.
$$

Time-dependent Dirichlet conditions, set by the reservoir concentrations, are imposed on each:

$$
C=C_1(t)\quad\text{on }\Gamma_1,
\qquad
C=C_2(t)\quad\text{on }\Gamma_2.
$$


No boundary condition is explicitly imposed on the top and bottom edges. In the finite-element weak formulation, an unconstrained boundary defaults to a homogeneous Neumann condition,


$$
\mathbf{J}\cdot\mathbf{n}=0
$$

which physically means the top and bottom edges are impermeable i.e no solute crosses them. Given the boundary and initial conditions specified here, this makes the problem effectively one-dimensional along $x$, even though it is solved on a two-dimensional mesh.

Because both reservoirs are assumed to be perfectly mixed, $C_1(t)$ and $C_2(t)$ are spatially uniform.


## 6. Initial conditions

The clay domain is initially solute-free:

$$
C(\mathbf{x},0)=0.
$$

and the reservoirs start at

$$
C_1(0)=\widehat C=0.02\ \mathrm{mol/m^3},
\qquad
C_2(0)=0,
$$

where $\widehat C$ denotes the reference initial (upstream) concentration.


The domain initial condition ($C=0$ everywhere) is incompatible with the boundary condition at $t=0$, which requires $C=\widehat C$ at $x=0$. This mismatch produces a steep concentration gradient near the left boundary during the first few time steps. Note that the stored plotting history records a transport rate of exactly zero at $t=0$, while the first nonzero left-boundary rate appears at the first computed time step, $t=\Delta t$.


## 7. Reservoir equations

The two reservoir concentrations evolve according to separate mass-balance ODEs,

$$
V_1\frac{dC_1}{dt}=\dot N_1,
\qquad
V_2\frac{dC_2}{dt}=\dot N_2,
$$

where $\dot N_1$ and $\dot N_2$ are computed by integrating the flux against the outward unit normal $\mathbf{n}$ of the clay domain at each interface:

$$
\dot N_1=B\int_{\Gamma_1}\mathbf{J}\cdot\mathbf{n}\,ds,
\qquad
\dot N_2=B\int_{\Gamma_2}\mathbf{J}\cdot\mathbf{n}\,ds.
$$


With this convention:

- $\dot N_1<0$ when reservoir 1 loses solute to the clay;
- $\dot N_2>0$ when solute leaves the clay and enters reservoir 2.

Because the domain is two-dimensional, the line integrals above carry units of $\mathrm{mol\,m^{-1}\,s^{-1}}$; multiplying by the thickness $B$ converts them into a total modeled molar transport rate, $[\dot N_1]=[\dot N_2]=\mathrm{mol\,s^{-1}}$. These quantities should therefore be called **molar transport rates** or **molar flow rates**, rather than local molar fluxes.


## 8. Domain solute inventory

The implemented model computes the domain solute inventory as

$$
N_{\mathrm{domain}}
=
B\int_\Omega C\,d\Omega.
$$

and the total modeled solute inventory as

$$
N_{\mathrm{total}}
=
C_1V_1
+
N_{\mathrm{domain}}
+
C_2V_2.
$$

For mass-conservation checks, the reference initial inventory is

$$
N_0=\widehat C V_1.
$$

i.e. the initial solute mass in reservoir 1, since the clay domain and reservoir 2 both start at zero. The absolute and relative mass-conservation residuals are then

$$
r_N=N_{\mathrm{total}}-N_0,
\qquad
r_{\mathrm{rel}}=\frac{r_N}{N_0}.
$$

($N_{\mathrm{domain}}$, $N_{\mathrm{total}}$, and $N_0$ carry units of mol; $r_{\mathrm{rel}}$ is dimensionless.)


## 9. Weak formulation

Let $q$ be a test function. Backward Euler time discretization gives

$$
\int_\Omega
\frac{\phi}{\Delta t}C^{n+1}q\,d\Omega
+
\int_\Omega
D_{\mathrm{eff}}
\nabla C^{n+1}\cdot\nabla q\,d\Omega
=
\int_\Omega
\frac{\phi}{\Delta t}C^nq\,d\Omega.
$$

The corresponding UFL forms are

```python
a = (
    (phi / dt) * C * q * ufl.dx
    + Deff * ufl.inner(ufl.grad(C), ufl.grad(q)) * ufl.dx
)

L = (phi / dt) * C0 * q * ufl.dx
```

## 10. Discretization

The domain is discretized using a structured $32\times32$ mesh of triangular cells:

```python
mesh.create_rectangle(
    points=((0.0, 0.0), (L0, H0)),
    n=(32, 32),
    cell_type=mesh.CellType.triangle,
)
```
The concentration is represented in a first-order continuous Lagrange space, $V=P_1$, and the projected flux field in a vector-valued first-order Lagrange space, $V_J=[P_1]^2$.


Because the gradient of a $P_1$ concentration field is cellwise constant, the flux used for ParaView visualization is instead obtained by an $L^2$ projection into the continuous vector space $V_J$: find $\mathbf{J}_h\in V_J$ such that, for all test functions $\mathbf{w}\in V_J$,

$$
\int_\Omega \mathbf{w}\cdot\mathbf{J}_h\,d\Omega
=
\int_\Omega \mathbf{w}\cdot\left(-D_{\mathrm{eff}}\nabla C_h\right)d\Omega.
$$


## 11. Boundary tagging and integration
The left and right facets, $\Gamma_1$ and $\Gamma_2$, are located using


```python
np.isclose(x[0], 0.0)
np.isclose(x[0], L0)
```
and assigned marker values 1 and 2, respectively. The tagged boundary measure is then defined as

```python
ds = ufl.Measure("ds", domain=msh, subdomain_data=mt)
```
where `mt` holds the facet tags, and the outward unit normal is obtained with

and the outward unit normal is
```python
n = ufl.FacetNormal(msh)
```
Each boundary's local transport-rate contribution is then assembled as

```python
ufl.dot(J(Cn), n) * ds(left_id)
ufl.dot(J(Cn), n) * ds(right_id)
```

## 12. Temporal Time discretization
The final run uses

$$
T=36000\ \mathrm{s}=10\ \mathrm{h},
\qquad
\Delta t=30\ \mathrm{s},
$$


corresponding to 1,200 accepted time steps.

Backward Euler is first-order accurate in time and numerically dissipative. Comparisons with larger time steps showed that the early-time transport rates are especially sensitive to temporal resolution, which motivated the smaller $\Delta t$ used in the final run. A formal convergence study for the final two-ODE formulation has not yet been reported.

## 13. PDE–ODE coupling

At each time step:

1. initialize the fixed-point guesses, $C_{1,\mathrm{iter}}$ and $C_{2,\mathrm{iter}}$, from the previously accepted reservoir concentrations;
2. apply them as Dirichlet boundary conditions;
3. solve the diffusion PDE;
4. integrate the left- and right-boundary transport rates, $\dot N_1$ and $\dot N_2$;
5. solve the two reservoir ODEs to obtain updated concentrations, $C_{1,\mathrm{new}}$ and $C_{2,\mathrm{new}}$;
6. compare these against the current guesses, $C_{1,\mathrm{iter}}$ and $C_{2,\mathrm{iter}}$;
7. if not converged, update the guesses and repeat from step 2;
8. once converged, accept the time step and update the stored domain field.

The fixed-point error is

$$
\varepsilon=\max\left(\left|C_{1,\mathrm{new}}-C_{1,\mathrm{iter}}\right|,\ \left|C_{2,\mathrm{new}}-C_{2,\mathrm{iter}}\right|\right)
$$

(with the same units as $C_1,C_2$: mol/m³). The iteration is accepted once

$$
\varepsilon<10^{-10},
$$

with a maximum of 25 fixed-point iterations allowed per time step.

## 14. Linear solver

The finite-element linear systems are solved through PETSc using a direct LU factorization — `ksp_type = preonly`, `pc_type = lu` — rather than an iterative Krylov method. Solver failures are configured to raise an error rather than fail silently.

## 15. Reservoir ODE solver

The coupled reservoir ODEs are integrated using SciPy's `solve_ivp(..., method="RK23")`, an explicit, adaptive-step solver, with relative tolerance $\mathrm{rtol}=10^{-8}$, absolute tolerance $\mathrm{atol}=10^{-12}$, and a maximum internal step of $\Delta t/2$.

Within each fixed-point iteration, the boundary transport rates are treated as constant over the current outer time interval; under that assumption, RK23 integrates an effectively constant right-hand side.

## 16. Selected concentration profiles

The implemented model samples $C(x,H_0/2,t)$ — the concentration along the horizontal midline — at 300 points in $x$, at each of five selected time steps:

| Step | Physical time |
|---|---|
| 1   | 30 s   |
| 10  | 5 min  |
| 20  | 10 min |
| 80  | 40 min |
| 300 | 2.5 h  |

These are modeled line profiles, not laboratory concentration measurements. The 2.5-hour profile is used to illustrate later-time evolution; a formal steady-state criterion has not been evaluated, so "near steady state" here is a qualitative description only, not a quantitative claim.

## 17. Numerical outputs

The solver writes:

- `ficks.xdmf` — modeled concentration field;
- `ficks_flux.xdmf` — projected modeled flux field;
- `history.csv` — reservoir concentrations, rates, domain amount, and residuals;
- `selected_concentration_profiles.csv` — selected line-profile values;
- Matplotlib PNG/PDF figures.


## 18. Numerical cross-checks performed

The following checks were discussed or incorporated:

**Sign convention.** The predicted transport-rate signs should satisfy $\dot N_1<0$ and $\dot N_2>0$ during left-to-right transport.

**Global amount diagnostic.** The model records the total modeled amount, $N_{\mathrm{total}}$, and its residual relative to the initial amount, $N_0$.

**Initial condition.** A zero-rate point is stored at $t=0$; the first evolved rate is evaluated at $t=30\ \mathrm{s}$ (i.e. $t=\Delta t$).

**Time-step sensitivity.** Earlier comparisons showed that larger time steps smooth and under-resolve the early transient; a smaller final step of $\Delta t=30\ \mathrm{s}$ was selected for the 10-hour run.

**Qualitative concentration behavior.** The predicted profiles remain monotonic between the upstream and downstream sides and broaden with time — qualitatively consistent with diffusive transport.

## 19. Verification not yet completed

The model has not yet undergone:

- comparison with a closed-form analytical solution;
- formal spatial-convergence analysis;
- formal temporal-convergence analysis for the final formulation;
- comparison with a published benchmark;
- code-to-code verification;
- experimental calibration;
- experimental validation.

The mass residual alone is not sufficient to validate the physical model.

## 20. Modeling assumptions and simplifications

The model assumes:

**Material and geometry**
- saturated, homogeneous kaolin;
- constant, isotropic material properties;
- two-dimensional geometry with constant thickness.

**Transport physics**
- Fickian diffusion;
- no advection;
- no concentration-dependent diffusivity;
- no electrochemical migration;
- no temperature variation.

**Chemistry**
- no chemical reaction;
- no H⁺ buffering;
- no mineral dissolution or precipitation;
- no sorption;
- no multicomponent coupling.

**Reservoirs and boundaries**
- perfectly mixed reservoirs;
- fixed reservoir volumes;
- uniform boundary concentrations;
- zero top/bottom normal flux.

## 21. Principal numerical conclusions

The numerical simulation predicts:

- a steep early concentration gradient near the upstream boundary;
- progressive spreading of the concentration profile through the domain;
- a later-time profile that becomes more nearly linear;
- a negative left-boundary transport rate and a positive right-boundary transport rate ($\dot N_1<0$, $\dot N_2>0$), consistent with the sign convention in §7;
- a rapid early transport transient followed by slower evolution.

These are simulated outcomes of the selected mathematical model and parameter set. They are not empirical findings.

## 22. Experimental status and future validation

The physical experiment could not be performed because of financial and logistical constraints.

Future work should:

- construct the through-diffusion apparatus;
- saturate and characterize the kaolin specimen;
- document the actual specimen geometry and porosity;
- monitor reservoir pH or concentration over time;
- convert measurements consistently between pH and H⁺ activity/concentration;
- estimate experimental uncertainty;
- calibrate $D_{\mathrm{eff}}$ against the measurements;
- compare measured and predicted reservoir histories;
- validate the concentration and rate predictions;
- revise the model to include chemical reactions if the measurements require them.