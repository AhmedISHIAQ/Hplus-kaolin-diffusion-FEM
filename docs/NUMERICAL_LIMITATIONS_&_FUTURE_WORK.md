# Limitations

-   No physical validation: The modeled through-diffusion experiment has not been performed.
-   No calibrated diffusivity: The current diffusion, porosity, and tortuosity values are prescribed numerical inputs.
-   Fickian transport only: The model assumes classical diffusion with no advection.
-   No chemical reactions: H⁺ buffering, mineral dissolution, precipitation, sorption, and speciation are omitted.
-   No electrochemical coupling: Charge balance and multicomponent ion transport are not represented.
-   Homogeneous medium: Porosity, tortuosity, and diffusivity are constant and isotropic.
-   Ideal reservoirs: Each reservoir is assumed to be perfectly mixed and constant in volume.
-   Two-dimensional representation: The physical volume is reconstructed by multiplying two-dimensional integrals by the thickness B.
-   Classical infinite propagation speed: The Fickian diffusion equation predicts a nonzero influence throughout the domain for any t>0.
-   Unverified inventory convention: The treatment of porosity in the domain inventory should be checked against the intended definition of C.
-   Serial profile extraction: The current concentration-line sampling routine is not fully MPI-parallel.
-   Approximate later-time label: “Near steady state” has been used descriptively but is not based on a formal convergence criterion.


# Future work

Priority future work includes:

-   perform the physical through-diffusion experiment;
record upstream and downstream pH/concentration histories;
calibrate $$D_eff$$ against measured data;
-   validate predicted concentration and transport histories;
document literature sources for all material parameters;
-   perform mesh and time-step convergence studies;
-   compare with an analytical or published numerical benchmark;
-   clarify whether domain inventory is B∫
Ω
	​

CdΩ or ϕB∫
Ω
	​

CdΩ;
-   add reaction, buffering, and multicomponent transport where required;
-   make line-profile extraction MPI-safe;
-   add automated unit and regression tests.