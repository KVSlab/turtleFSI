# ROM Package Layout

## Reference-flag reproduction and time prediction

For the serial CG2/CG2/CG1 `TF_fsi_reference` runs, a complete baseline check is:

```bash
python -m turtleFSI.cases.validate_flag_rom \
  --run-dir simulations/aero/flag-geometry-tests.NgbVfy/results/baseline \
  --output-dir simulations/aero/flag-geometry-tests.NgbVfy/rom-new-validation
```

The output directory must be new. Defaults train on snapshots through 3.01 s,
reproduce 2.01--3.01 s, and predict 3.01--3.51 s. Each rollout is initialized
once from its starting FOM state; later FOM states are only comparison data.
Supremizers and POD modes use training snapshots only. The basis has at most
24 modes each for displacement, velocity, pressure, and supremizer enrichment.
Errors are reported separately as Euclidean coefficient norms per field,
using active fluid pressure DoFs. `tip_comparison.png` shows the displacement
probe; NPZ outputs retain full reconstructed/reference vectors and timestamps.
The summary records convergence and minimum deformation Jacobians at every
reduced timestep. An unsuccessful Newton solve stops the experiment and saves
partial results, so inspect `completed` in the summary.

Each phase also saves and reports its best constrained-basis projection error.
This uses held-out states only for diagnosis, never for basis training or
rollout updates. `--modes N` caps each of the four POD blocks; optional
`--displacement-modes`, `--velocity-modes`, `--pressure-modes`, and
`--supremizer-modes` allow separate caps. `--basis PATH` reuses a trained basis.
`effective_basis.npz` records the actual constrained trial/test basis.

For an unseen-geometry check, supply `--training-run-dir RUN1 RUN2` while
`--run-dir` identifies the held-out evaluation case. Each training run is
truncated at `--train-end` before supremizer construction. With the current
baseline/longer-flag training pair and a larger-circle holdout, this is radius
extrapolation. Both phases are initialized once from the held-out FOM state;
this is not yet a cold-start parameter prediction.

`--kinematic-enrichment` is an experimental diagnostic that transports the
continuous displacement modes into the velocity space, preserves the existing
span, enforces homogeneous increments, and adds only independent columns.
It is opt-in: the first baseline experiment improved reproduction but worsened
held-out displacement prediction, so it is not a validated default fix.

This driver uses the saved FOM timestep (0.01 s), including substeps between
0.05 s snapshot samples. Do not treat snapshot spacing as the FOM timestep.
It assembles full-order FEniCS operators for projection; this is an accuracy
experiment, not an offline/online speedup demonstration or a parameter holdout.

Replay prefers the checkpoint adjacent to copied snapshots over the original
machine's absolute path. The ROM builder now calls `setup_reference_geometry`
before assembling forms; reference-flag replay reads the saved shape field,
checks the reference mesh/topology and parameter-dependent coordinates, and
avoids rewriting FOM outputs. Offline supremizers include the shape Jacobian
through the same transformed incompressibility integrand as the fluid forms.

For normal use, start with the two case-level files:

- `pod.py`: trains a generic POD/data-fit model from any compatible `.npz`
  snapshot bundle.
- `pod_galerkin.py`: builds a monolithic POD basis from
  `ROM/dvp_snapshots.npz` and can optionally run a projected FEniCS
  POD-Galerkin step or rollout. It also exposes the Ballarin/Rozza-style
  blocked/supremizer operator-bundle workflow.

The installed command names are:

- `turtleFSI-pod`
- `turtleFSI-pod-galerkin`

The private folders are implementation details:

- `_pod`: helpers for POD/data-fit models.
- `_pod_galerkin`: helpers for projected POD-Galerkin models.
- `_shared`: POD math used by both workflows.

Inside `_pod_galerkin`, `reduced_solvers.py` contains the reduced Newton,
linear projection, and rollout math. This keeps `turtleFSI/rom/pod_galerkin.py`
as the only `pod_galerkin.py` file and makes it the public case-level workflow
and CLI entry point.

Case-specific exporters, snapshot preparation, and plotting tools live in
`turtleFSI/cases`, not in this package.

## Explicit time and parameter snapshot matrices

### Identity-map diagnostic for a saved reference-flag case

To isolate the fixed geometry transformation, reuse that case's single-geometry
basis and snapshots on its saved physical initial mesh:

```bash
python -m turtleFSI.cases.validate_flag_rom \
  --run-dir simulations/aero/flag-training-completed/results/radius_00_length_01 \
  --basis /path/to/single-case/basis/monolithic_pod_basis.npz \
  --output-dir /path/to/new/identity-comparison \
  --identity-map --train-end 14.01 --reproduction-only \
  --reproduction-start 10.01 --prediction-end 14.01
```

The computational mesh becomes the saved mapped mesh, with zero shape
displacement (`G=I`, `K=1`). Cellwise mixed DoF ordering, boundary constraints,
and active pressure DoFs must match before coefficient vectors can be reused.
The original cellwise geometry determinant remains only in the ALE extension
diffusivity `1/(J*K_original)`, preserving the equation used for the saved FOM.
This is a coordinate-equivalence test, not a newly generated identity-map FOM
with a different extension equation. It performs no new POD training.

Before rollout, full residuals and projected Jacobians are compared at identical
perturbed states sampled from the saved trajectory. A relative difference above
`1e-8` stops the diagnostic. Measurements are written to
`identity_operator_check.json`; rollout results use the standard validation
format. Compare against the original single-case rollout using the same basis,
initial time, end time, and timestep. The physical deformation `F` remains
time-dependent; only the fixed geometry map becomes identity.

The blocked basis builder accepts one snapshot bundle per training parameter:

```bash
python -m turtleFSI.rom._pod_galerkin.build_monolithic_pod_basis \
  --snapshots run_mu1/ROM/dvp_supremizers.npz run_mu2/ROM/dvp_supremizers.npz \
  --output-dir analysis/parameter_basis \
  --blocked-dvp --pressure-treatment supremizer
```

For each parameter index `i`, the builder constructs `Si_d`, `Si_u`, `Si_p`,
and `Si_s` with saved times as columns. It then builds `S_d`, `S_u`, `S_p`,
and `S_s` by concatenating the corresponding parameter blocks horizontally,
in command-line order. These combined matrices supply the field PODs.
`--start-index` and `--stop-index` select a time window separately in each
input bundle; parameters can have different numbers of snapshots.

The output `snapshot_matrices.npz` contains `S1_d`, `S2_d`, etc., the combined
`S_d` through `S_s`, field DoF maps, `parameter_sources`, and
`parameter_column_offsets`. It also saves `snapshot_indices_1`, etc., and
`time_1`, etc., when input time arrays exist. Indices are 1-based for parameter
names and 0-based for snapshot indices/column offsets. Source paths identify
parameter runs; the builder does not infer numerical parameter values from
filenames. The basis summary records each block's source, shape, and columns.

Saved matrices contain raw snapshots, before pressure gauge adjustment and
POD centering. With one bundle, each combined matrix equals its single
parameter block. Supremizer matrices are omitted if no input has them and
enrichment is disabled; enrichment requires them for every parameter.

For blocked POD, `monolithic_pod_basis.npz` also saves the reduced basis
matrices explicitly as `Z_d`, `Z_u`, `Z_s`, `Z_u_s`, and `Z_p`. All have
full mixed-space rows, with zeros outside their field's DoFs.
`Z_u_s = [Z_u | Z_s]` and `basis = [Z_d | Z_u_s | Z_p]`.
`Z_s` contains the final supremizer modes after orthogonalization against
velocity modes. Without enrichment, it has zero columns and `Z_u_s = Z_u`.
These arrays are saved when the basis builder is rerun; existing output
files are not modified automatically.

All bundles must use the same reference mesh, finite element space, and DoF
ordering. The builder checks mixed dimensions and field DoF arrays, but those
checks alone cannot establish that meshes have identical geometry.

## Explicit equation (6) operators

Add `--equation-6-blocks` when calling the modules
`turtleFSI.rom._pod_galerkin.build_ballarin_operator_bundle`,
`turtleFSI.rom._pod_galerkin.run_fenics_pod_galerkin_step`, or
`turtleFSI.rom._pod_galerkin.run_fenics_pod_galerkin_rollout` directly.
This requires a blocked basis and the equation (4) model: implicit Euler
(`theta=1`), `extrapolation_sub_type=paper`, and a supported linear-elastic solid.

`equation_6.py` explicitly builds `A_F_N = Z_u_s.T @ A_F @ Z_u_s`
and the other named projections, then constructs `J_dd_N`, `J_du_N`,
`J_ud_N`, `J_uu_N`, `J_up_N`, `J_pd_N`, and `J_pu_N` with equation (4)'s
signs. The combined order is displacement, enriched velocity, pressure.
Mass terms already include time-step factors; `C_F_u_N` is the Newton
convection derivative. These are Euclidean Galerkin projections.

The physical block sum is checked against the projection of the full raw
Jacobian. A mismatch raises an error, including when additional loads or
supports are missing from the paper forms. Dirichlet row replacement and
inactive-row repairs are represented separately by
`equation_6_boundary_correction_N`, rather than added to every physical term.
`J_N` is the physical block sum plus this correction, checked with the same
basis and boundary treatment as the aggregate projection.

The operator bundle saves all these named arrays directly in its output NPZ;
`load_reduced_operator_bundle(...).named_equation_6_operators` exposes them.
The Newton step saves `equation_6_operators.npz` evaluated at its returned
state. The rollout saves `equation_6/step_00001.npz`, etc., at each returned
time-step state. Operators are rebuilt at every requested Jacobian update.
This inspection mode still assembles full-order FEniCS matrices and performs
an extra aggregate check; it is not a hyper-reduced online implementation.

## Calling a Case

For a POD/data-fit model, point the command at the snapshot bundle for the
case:

```bash
turtleFSI-pod \
  --snapshots simulations/arterial-stenosis-200ms-stvk/ROM/dvp_snapshots.npz \
  --output-dir simulations/arterial-stenosis-200ms-stvk/analysis/data-fit
```

For a POD-Galerkin model, point the command at the simulation run directory
that contains `ROM/dvp_snapshots.npz`:

```bash
turtleFSI-pod-galerkin \
  --run-dir simulations/arterial-stenosis-200ms-stvk \
  --output-dir simulations/arterial-stenosis-200ms-stvk/analysis
```

These commands are problem-agnostic as long as the case writes the expected
snapshot arrays.

Supremizers are computed **after** the simulation. Enable only
`--rom-export-dvp true` during the full-order run. Then, in the activated
FEniCS environment, run:

```bash
python -m turtleFSI.rom.compute_supremizers \
  --run-dir simulations/arterial-stenosis-200ms-stvk
```

This solves `X s_i = B_F(d_i)^T p_i` from each saved displacement and pressure
snapshot using `Mesh/mesh.h5` and `Checkpoint/default_variables.json`.
It writes `ROM/dvp_supremizers.npz` and its summary, preserving the original
bundle. Keep these mesh and settings files with the snapshots. The command
currently supports serial snapshots and must run on one MPI rank; older
bundles without MPI-size metadata are assumed serial and checked against the
reconstructed field DoF maps. Use `--input` and `--output` for custom paths.
The removed `--rom-export-supremizer` simulation flag is no longer needed.

For a Ballarin/Rozza-style reduced-operator workflow, build the blocked
basis and reduced operator bundle from the enriched snapshots:

```bash
turtleFSI-pod-galerkin \
  --run-dir simulations/arterial-stenosis-200ms-stvk \
  --snapshots simulations/arterial-stenosis-200ms-stvk/ROM/dvp_supremizers.npz \
  --blocked-dvp \
  --pressure-treatment supremizer \
  --scheme ballarin-monolithic \
  --action build-operators \
  --reference-index 0 \
  --output-dir simulations/arterial-stenosis-200ms-stvk/analysis
```

Then solve the saved reduced system online:

```bash
turtleFSI-pod-galerkin \
  --run-dir simulations/arterial-stenosis-200ms-stvk \
  --snapshots simulations/arterial-stenosis-200ms-stvk/ROM/dvp_supremizers.npz \
  --blocked-dvp \
  --pressure-treatment supremizer \
  --scheme ballarin-monolithic \
  --action operator-solve \
  --operator-bundle simulations/arterial-stenosis-200ms-stvk/analysis/operators/reduced_operators.npz \
  --output-dir simulations/arterial-stenosis-200ms-stvk/analysis
```

The online solve uses the saved reduced system without full FEniCS assembly.
The current bundle builder projects the monolithic residual/Jacobian linearized
at a reference snapshot. Full paper-level nonlinear/parametric EIM
hyper-reduction is still a separate extension.

## Physical flag-pole comparison plots

For a new `TF_fsi` run, `--action validate-rollout` now generates physical
comparison plots and an MP4 in `OUTPUT_DIR/physical_comparison/` unless
`--skip-validation-plots` is supplied. `--plot-fps` sets the animation frame rate.
The FOM solver, POD basis, and ROM rollout are unchanged by this postprocessing.

To plot an already completed run without solving again:

```bash
python -m turtleFSI.cases.plots.compare_fsi --run-dir "$run_dir" --fps 20
xdg-open "$run_dir/analysis/physical_comparison/animation.html"
```

The command discovers `analysis/validation_rollout/fenics_pod_galerkin_rollout.npz`
or `analysis/rollout/fenics_pod_galerkin_rollout.npz`. If both exist, or the rollout
is elsewhere, pass `--rollout PATH`. Use `--output-dir PATH` to select the output
folder, and `--static-only` to skip video generation. The source run must retain
`Checkpoint/default_variables.json` and `Mesh/mesh.h5` with the original DOF
ordering. Activate the FEniCS environment first; MP4 generation also needs ffmpeg.

Outputs include physical error CSV/NPZ data, `time_comparison.png` and PDF,
`spatial_comparison.png`, `fom_rom_animation.mp4`, a looping HTML player, and
metadata recording source paths, time range, probe location, and frame count.
The spatial L2 errors integrate the FE fields on solid/fluid domains. A zero
reference norm gives an undefined relative error (NaN in CSV/NPZ).

This command supports compatible **2-D triangular FSI cases**, including different
flag materials, mesh sizes, time windows, and snapshot counts. Three-field DVP
and four-field biharmonic mixed spaces are supported. Flag tip coordinates come
from the saved geometry; other 2-D FSI geometries require `--tip X Y` to select a
reference probe point. It does not replace the separate 3-D surface plotter.

Both upper panels update geometry and speed at every saved state. **No displacement
amplification is applied in any panel.** A nearly stationary physical solution
will still appear nearly stationary. Each saved state gets one frame; irregular
simulation time spacing therefore gives nonuniform physical playback speed.
