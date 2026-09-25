# PoCA-Z uncertainty calibration (development scenes)

This plan precedes implementation. Use only U, L and double-block development
scenes; the previously inspected T scene is not an independent test.

For each paired ideal/noisy event, estimate its Z instability from the *noisy
measured hits alone* using 32 fresh Gaussian hit perturbations at the known
detector resolution. Never use the ideal PoCA or target mask in this estimate.
Compare the predicted uncertainty with the absolute movement between noisy and
ideal algorithmic PoCA estimates, and with Z-boundary gate flips. The latter
is a repeatability target, not true scattering-point error.

Primary checks: median and P90 of `abs(delta_z) / predicted_sigma_z`, 68% and
90% nominal interval coverage, and boundary-flip rates across preregistered
uncertainty tertiles. Keep sigma=0 as a no-change control. Report per scene,
input count and hit-resolution level, including failures/undefined estimates.
Do not tune a reconstruction threshold or claim imaging improvement here.
