# Geometry variants of the flag/circle benchmark

The implementation follows the distinction in `paper/galerkin-paper.pdf`,
Sections 2.1 and 2.3, between the fixed shape map T_mu and subsequent physical
ALE motion. Section 4.2 uses an affine leaflet-length map; it does not provide
a circle/flag-radius map. The angular/solid/harmonic construction here is our
implementation choice, not a formula claimed to come from that paper.

## Fixed geometry map

`TF_fsi_geometry` loads and labels the original `TF_fsi` mesh at the baseline
geometry, then changes only vertex coordinates. It keeps channel dimensions,
circle center, flag thickness, connectivity and cell/facet identities fixed.

For the flag, with eta = y - c_y, define

    root0(eta) = c_x + sqrt(R0^2 - eta^2)
    root(eta)  = c_x + sqrt(R^2  - eta^2)
    tip0 = c_x + R0 + length0
    tip  = c_x + R  + length
    x_new = root + (x - root0)/(tip0 - root0) * (tip - root)
    y_new = y

Thus the curved root follows the circle and the flag height remains 0.02 m.
The exposed circle arc is mapped by rescaling its angular interval between
the two flag-attachment endpoints. The outer channel boundary stays fixed.
The default map is `radius_then_length_v2`. It has two stages:

1. Change the radius while holding flag length at 0.35 m. Extend the prescribed
   circle/solid displacement into the fluid with a P1 Laplace solve.
2. Change flag length by an explicit horizontal deformation of those vertices.
   Between the curved root and the old tip, the horizontal displacement grows
   linearly from zero to the length change. Downstream, it decreases linearly
   to zero at the outlet. Above and below the flag, a linear vertical taper
   reduces it to zero at the channel walls. Vertical coordinates stay fixed
   during this second stage.

For stage 2, with `(x,y)` the radius-deformed coordinates, define:

    anchor(y) = c_x + sqrt(R^2 - min((y-c_y)^2, (f_H/2)^2))
    old_tip = c_x + R + 0.35
    weight_x = max(0, (x-anchor)/(old_tip-anchor))  if x <= old_tip
             = (L-x)/(L-old_tip)                 otherwise
    weight_y = min(1, y/(c_y-f_H/2), (H-y)/(H-c_y-f_H/2))
    x_new = x + (length-0.35) * weight_x * weight_y
    y_new = y

The circle, solid and wall vertex constraints are set exactly after this stage.
T is the P1 interpolant of the final vertex positions on the original triangles;
G and K are computed from these final positions. The analytic deformation is
not globally affine, and we do not claim an affine parameter separation of the
operators. Its discrete map is affine on each reference triangle.

This avoids concentrating a large length change in the fluid triangles beside
the flag tip. It is our construction, not an implementation of an unspecified
formula from the paper. Both stages are separate from the time-dependent
`paper` extension used in the FSI simulation. No remeshing is performed.

The original single harmonic extension remains available as `harmonic_v1`.
New geometry reports record `mapping_version`. ROM replay and checkpoint
restart select that saved version; reports without a version use v1,
preserving existing snapshots.
New training runs should use v2 consistently rather than mixing map versions.

Each target mesh is rejected if a triangle's signed area ratio is below 0.2
or its normalized triangle quality is below 0.02. These are practical guards,
not a proof of simulation accuracy for arbitrary geometry changes. Circle
vertices lie on the requested circle; the mesh remains a polygonal
approximation between vertices, as in the original linear geometry mesh.

## Validation and use

Activate the existing FEniCS environment and run from the project root:

```bash
export PYTHONPATH="$PWD"
python -m turtleFSI.cases.validate_flag_geometry --output-dir geometry-validation
```

This validates baseline (R=0.05, length=0.35), longer flag (0.05, 0.3675),
and larger circle (0.0525, 0.35). It checks circle vertices, tip position,
unchanged solid y coordinates, identical connectivity, tags and mixed-space
cell DoF maps. All use CG2 displacement/velocity and CG1 pressure. There are
21,749 mixed DoFs: 10,216 displacement, 10,216 velocity and 1,317 pressure.
The previous POD summary's 1,266 pressure DoFs counted only active fluid
pressure DoFs, not the whole mixed pressure subspace.

To check a 25-case grid with lengths 0.35–0.50 m and radii 0.05–0.075 m:

```bash
python -m turtleFSI.cases.validate_flag_geometry \
  --output-dir geometry-validation-wide \
  --radii 0.05 0.05625 0.0625 0.06875 0.075 \
  --lengths 0.35 0.3875 0.425 0.4625 0.5
```

This checks sampled initial geometries, not every parameter in the continuous
rectangle or their stability during 30-second FSI runs. Every actual run still
checks its own initial map, and the reference solver monitors J and JK in time.

To validate then run three resource-limited short server tests:

```bash
bash turtleFSI/cases/run_flag_geometry_tests.sh /home/ml49985
```

The script requires an activated environment, user systemd with delegated
cgroup-v2 CPU/memory controllers, `/usr/bin/time`, and `timeout`. It refuses
to start a solver if the memory/swap/CPU limits cannot be verified. Each case
uses one MPI process, one computational thread, 2 GiB memory maximum, no swap,
one CPU quota, and a ten-minute timeout. The three independent cases run
concurrently only after geometry validation succeeds. Output is a new unique
directory; old runs are not overwritten. Per-case stdout and resource reports
are retained. This is a short smoke test, not a long-run memory guarantee or
an enforced disk quota. The existing in-memory NPZ snapshot exporter is kept
for these small tests only; streaming export is still needed for long runs.
The current solver advances once past the requested T, so T=0.1 ends at 0.11.
Check logs for Newton convergence in addition to the process exit status.

Only R and f_L are variable. One MPI rank per geometry is enforced. Other
material, solver and time settings remain identical across cases. The problem
defaults to the pilot's LinearElastic material, theta=1 and paper extension,
and supplies the required fsi_id=5. FSI labels are carried from the baseline
mesh instead of recomputed using tolerances on the deformed geometry.

## Common-reference full-order simulation

`TF_fsi_reference` uses the same shape construction but solves on the unchanged
baseline mesh. `build_flag_reference()` supplies a vector CG1 function
`shape_displacement` g, the UFL expressions T = X + g, G = I + grad(g), and
K = det(G), plus independent per-triangle G/K arrays. Displacement and velocity
remain CG2 and pressure CG1 when the degree options below are used. All cases
therefore share reference coordinates, topology, tags, and reference FE bases.

The solver's `setup_reference_geometry` hook runs after mesh creation and before
variational assembly. It supplies g to the existing transformed fluid, solid,
and paper-extension forms. Physical displacement starts at zero; the three
configurations are X, T(X), and T(X) + d(X,t). In the fluid, d is ALE mesh
displacement. The computational mesh is never moved during this procedure.

From an activated project FEniCS environment, a short longer-flag run is:

```bash
python -m turtleFSI.run_turtle --problem TF_fsi_reference \
  --end-time 0.1 --time-step 0.01 --d-deg 2 --v-deg 2 --p-deg 1 \
  --folder flag-reference-results --sub-folder longer-flag \
  --new-arguments R=0.05 f_L=0.3675
```

The case inherits LinearElastic, theta=1, and laplace/paper extension defaults.
To validate the geometry and launch all three short reference cases concurrently
with the server runner's verified resource limits, use:

```bash
bash turtleFSI/cases/run_flag_geometry_tests.sh "$HOME" TF_fsi_reference
```

The runner requires user systemd scopes with delegated CPU/memory controllers.
It exports a snapshot every timestep, writes separate logs, and waits for all
three processes. Each case runs serially; do not launch this runner with MPI.

After the reference runs finish, verify their saved representations with:

```bash
python -m turtleFSI.cases.validate_flag_snapshots /path/to/flag-geometry-tests.ID
```

This read-only checker expects the runner's `results/{baseline,longer_flag,
larger_circle}` layout and serial CG2/CG2/CG1 spaces. It compares the saved
reference coordinates, connectivity, cell/facet tags, reconstructed cell DoF
maps, exported field indices and active pressure indices. It also checks
snapshot dimensions, finite entries, matching increasing times/counters,
interface adjacency, and consistency of the saved shape field, mapped
coordinates, circle radius, flag tip, and per-cell G/K. A mismatch exits with
status 1 and the failing case. This does not test long-time physical accuracy,
current-configuration Jacobians J, or ROM operator loading.

It saves the reference mesh as `Mesh/mesh.h5`, g as
`Mesh/shape_displacement.h5`, and reference/mapped coordinates, tags, and
cellwise G/K as `Mesh/geometry_map.npz`. Standard solution output is on the
reference mesh: reconstruct physical positions as X + g + d when visualizing.
Tip displacement is evaluated at the baseline tip; drag/lift use transformed
surface traction and the fluid-side interface trace.

`tests/test_flag_reference.py` checks the UFL map gradients against triangle
Jacobians, P1/P2 mass and diffusion operators against mapped-mesh assembly,
and two full FSI timesteps on both domains for baseline and changed geometry.
For the last comparison the direct mapped-domain extension retains the
paper's diffusivity 1/(J*K). Running `TF_fsi_geometry` with its default K=1
would change that extension equation and is not an equivalent comparison.

These short checks establish the coordinate transformation and solver wiring;
they do not reproduce the paper's numerical examples or establish long-time
accuracy of this flag benchmark.

## Geometry-aware ROM work still required

### 30-second v2 training campaign

To run exactly the 25 combinations in the mesh overview, pass `grid25` as
the seventh runner argument:

```bash
bash turtleFSI/cases/run_flag_training.sh /home/ml49985 3 30 0.01 12h 5 grid25
```

This uses radii 0.05, 0.05625, 0.0625, 0.06875, 0.075 m crossed with lengths
0.35, 0.3875, 0.425, 0.4625, 0.50 m. All 25 use the base timestep and snapshot
stride. This plan adds no refinement cases and reserves no held-out geometries;
select a train/test split before POD training. Without the final argument, the
original `training16` plan described below remains the default.

`run_flag_training.sh` prepares and validates these geometries before launching:

* 12 training cases: radii 0.05, 0.0625, 0.075 m crossed with flag lengths
  0.35, 0.40, 0.45, 0.50 m.
* Two held-out cases: (radius, length) = (0.05625, 0.425) and (0.06875, 0.475).
* Two time-step checks: baseline and largest geometry, each with dt=0.005
  instead of 0.01. These are excluded from the training list.

After updating the code on Aero, run in its activated turtleFSI environment:

```bash
cd /home/ml49985/turtleFSI
bash turtleFSI/cases/run_flag_training.sh /home/ml49985 3 30 0.01 12h 5
```

Arguments are output parent, concurrent cases, end time, base time step,
per-case timeout, and base snapshot stride. Defaults match the command above.
The runner uses batches of at most three independent serial cases, 2 GiB RAM
and one CPU quota per case, no swap, and new unique output directories.
A failed batch stops further launches. Checkpoints are saved every 500 steps.
Use a persistent terminal session or nohup so an SSH disconnect does not stop
the campaign. The runner does not automatically resume interrupted campaigns.

For a detached launch, after activating the environment:

```bash
launch_log=$(mktemp /home/ml49985/flag-training-launch.XXXXXX.log)
nohup bash turtleFSI/cases/run_flag_training.sh \
  /home/ml49985 3 30 0.01 12h 5 > "$launch_log" 2>&1 < /dev/null &
echo "PID: $!; launch log: $launch_log"
```

The first line of that log identifies the unique result directory. At completion,
`campaign_validation.json` contains representation checks, min J/JK, and RMS
differences for tip displacement, drag, and lift between dt and dt/2 runs.
Comparison uses fine-history interpolation at coarse timestamps through T.
It is a sensitivity diagnostic, not an automatic accuracy certification or
a spatial convergence test. The solver's existing one-step overshoot is checked
explicitly. Every expected history row and exported snapshot time must exist.

`training_runs.txt` and `heldout_runs.txt` identify the disjoint data splits.
Review the time-step comparisons and trajectories before training the ROM.
Preserve `campaign.json` when transferring outputs back to hlu. To validate
again (also regenerating the split lists with the current machine's paths):

```bash
python -m turtleFSI.cases.flag_campaign validate /path/to/flag-training.ID
```

No plotting package is needed on Aero. Code must be synchronized before launch;
the scripts prepared on hlu are not automatically available on Aero.

### Longer full-order validation runs

The runner accepts optional arguments after the problem name: end time, time
step, per-case wall timeout, and snapshot stride. For a first run beyond the
2-second inlet ramp:

```bash
bash turtleFSI/cases/run_flag_geometry_tests.sh "$HOME" TF_fsi_reference 5 0.01 2h 5
```

Each reference case appends `reference_history.csv` every solved timestep:
time, tip displacement components, drag, lift, minimum J, and minimum JK.
The CG2/CG1 triangle monitor recovers the quadratic J in DG2 and evaluates
vertex, edge, and interior extrema, up to assembly roundoff. It stops on a
non-positive minimum and writes the failing J/JK. Positive determinants do not
guarantee absence of global self-contact or accurate spatial resolution.
The histories survive an early stop. The solver's existing end-time behavior
may add a timestep. A 5-second run is an initial validation interval, not a
claim of periodic or time-converged behavior.

After all cases finish:

```bash
python -m turtleFSI.cases.validate_flag_snapshots /path/to/new/run
python -m turtleFSI.cases.plot_flag_validation /path/to/new/run
```

The plot command writes `validation_plots/histories.png` and `summary.json`.
Inspect convergence logs as well: producing a plot does not certify successful
completion. Compare against a separate dt=0.005 run before selecting training
data. Snapshot accumulation and compressed rewrites remain in memory; the
stride of five limits their cost for this initial validation interval.

`TF_fsi_geometry` runs full simulations on each parameter's physical initial mesh. Their
displacement is relative to that initial mesh; it does not include the fixed
shape deformation. Do not also pass this same map as `shape_displacement`
to these runs: that would count the geometry transformation twice.

`Mesh/geometry_map.npz` stores reference/mapped vertices, connectivity and tags.
For each triangle, the affine Jacobian is G = B_mapped * inverse(B_reference),
where B has columns (x1-x0, x2-x0), and K = det(G). The common vertex and
cell-DoF identity provides a discrete reference pullback for the nodal fields.
However, equal array sizes alone do not make the current multi-geometry ROM
correct. A common-reference ROM needs consistent pullbacks and geometry-aware
operators/metrics, using G and K as in Section 2.3. The current POD builder
assumes a common reference mesh and should not be fed these bundles as if
the physical meshes were identical. That ROM integration is not implemented
by the mapped-mesh generation feature. `TF_fsi_reference` now supplies the
common-reference full-order path, but the ROM operator reconstruction must
also load g and preserve its parameter dependence. Do not assume a ROM reader
automatically consumes the new shape-displacement artifact.

## Optional adaptive time stepping

`--adaptive-time-step true` enables convergence-based time stepping for fresh,
serial `TF_fsi_reference` runs with theta=1, CG2 displacement, and laplace/paper
extension. Fixed-step mode remains the default. Adaptive restart is not yet
supported; launch a fresh case in a new output directory.

Failed Newton solves (including iteration-limit exhaustion), non-finite fields,
and non-positive deformation determinants reject a trial. All three solution
history vectors and the inlet time are restored, dt is halved, and the step is
retried. The timestep Constant and Jacobian are updated for every attempt.
Three accepted steps using at most five Newton iterations increase the proposed
dt by 25%, capped by `--adaptive-dt-max` (the initial dt if omitted). Failure at
`--adaptive-dt-min` stops the run explicitly. Steps shortened to land on an
output or end time may be smaller than that minimum; they cannot be retried
with a still smaller dt. `adaptive_steps.csv` records every accepted/rejected
attempt. Rejected states never enter history, snapshots, or visualization.

In this mode, ROM and visualization output both follow physical times set by
`--adaptive-output-start` (default 0.01 s) and `--adaptive-output-interval`
(default 0.05 s). Their step-count strides are ignored. Checkpoints still use
accepted-step counts. The solver stops at the requested end time without the
legacy extra step: request 30.01 s to match the old 601-snapshot grid.

Example after activating the FEniCS environment, from the repository root:

```bash
python -c 'from turtleFSI.run_turtle import main; main()' \
  --problem TF_fsi_reference --end-time 30.01 --time-step 0.01 \
  --theta 1.0 --extrapolation laplace --extrapolation-sub-type paper \
  --d-deg 2 --v-deg 2 --p-deg 1 --checkpoint-step 500 \
  --rom-export-dvp true --adaptive-time-step true \
  --adaptive-dt-min 0.00015625 --adaptive-dt-max 0.01 \
  --adaptive-output-start 0.01 --adaptive-output-interval 0.05 \
  --folder /path/to/new/adaptive-results --sub-folder radius_04_length_04 \
  --new-arguments R=0.075 f_L=0.5 fsi_id=5
```

This controls nonlinear convergence, not temporal truncation error. It does not
add a Newton line search or change the residual/Jacobian integration settings.
It is not a guarantee that the failing geometries will complete. Check solution
accuracy separately. Existing fixed-step campaign/replay validators may assume
uniform solver counters; adaptive snapshot times are uniform but their counters
are not. The existing campaign launcher is not changed to enable this option.
