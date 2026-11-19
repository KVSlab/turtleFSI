# POD-Galerkin ROM Implementation Plan

## Reference

Local source paper:

```text
paper/galerkin-paper.pdf
```

Ballarin and Rozza, "POD-Galerkin monolithic reduced order models for
parametrized fluid-structure interaction problems", International Journal for
Numerical Methods in Fluids, 2016.

The target is a monolithic POD-Galerkin FSI ROM with global displacement,
global velocity, fluid pressure, supremizer stabilization, and an offline-online
efficient reduced system.

## Main Problems With the Current Code

The current code is useful, but it should not be described as a complete
Ballarin/Rozza implementation yet.

1. **The projected-FEniCS path is a correctness/debug path, not an efficient
   online ROM.**

   `turtleFSI/rom/_pod_galerkin/run_fenics_pod_galerkin_step.py` and
   `run_fenics_pod_galerkin_rollout.py` reconstruct a full state and evaluate:

   ```text
   R_N(a) = Phi.T R(mean + Phi a)
   J_N(a) = Phi.T J(mean + Phi a) Phi
   ```

   This is mathematically intrusive Galerkin projection, but it still assembles
   full FEniCS residuals and Jacobians online. The paper requires offline-online
   decomposition and empirical interpolation/hyper-reduction for the expensive
   nonlinear and parameter-dependent terms.

2. **The Ballarin-style branch is only partially online-ready.**

   `offline_online.py`, `build_ballarin_operator_bundle.py`, and the
   operator-solve scripts can save and solve reduced linearized operator
   bundles. That is not the full nonlinear time-marching Ballarin/Rozza model.
   The missing layer is a reduced block evolution system with named monolithic
   operator terms and nonlinear/hyper-reduced online evaluation.

3. **Workflow complexity hides method problems.**

   The public wrapper `turtleFSI/rom/pod_galerkin.py` has many combinations of:
   `--action`, `--scheme`, `--blocked-dvp`, pressure treatment, pressure gauge,
   train/test split, linear solver, mode caps, and plot settings. Users can
   easily run something that is valid syntactically but not the intended
   validation experiment.

4. **Mode counts are easy to misunderstand.**

   The field mode flags are caps when combined with an energy threshold. A run
   asking for many modes can still retain very few modes if the energy threshold
   is reached early. This caused undersized bases in the 0.2 s arterial stenosis
   validation. Use `--exact-modes` for experiments where requested caps should
   be treated as exact rank requests subject only to snapshot rank.

5. **Supremizers address pressure stability, not long-horizon accuracy.**

   Supremizer enrichment is necessary for velocity-pressure compatibility, but
   it does not guarantee a stable 40-step nonlinear rollout. The current
   validation showed large velocity error despite pressure stabilization, so the
   next fixes need to target reduced dynamics, conditioning, and nonlinear
   operator treatment.

6. **Diagnostics are not yet strong enough.**

   Existing summaries report residuals and some field errors, but paper-level
   development needs reduced divergence/incompressibility residuals, inf-sup
   surrogate singular values, per-field interior errors, Jacobian condition
   history, mode-count provenance, and timing that separates FEniCS assembly
   from reduced solves.

7. **The implementation is still mostly single-trajectory.**

   The paper is parametrized. A closer reproduction needs parameter metadata,
   multiple training parameter samples, and independent online parameter tests.

## What Is Implemented

The repository now has the core scaffolding for a serious intrusive ROM:

- exact mixed FEniCS DoF snapshot export in
  `turtleFSI/cases/snapshots/dvp_export.py`;
- offline supremizer computation with `python -m turtleFSI.rom.compute_supremizers --run-dir RUN_DIR`;
- fluid-pressure DoF extraction for pressure POD spaces;
- blocked displacement, velocity, pressure, and supremizer basis construction
  in `turtleFSI/rom/_pod_galerkin/build_monolithic_pod_basis.py`;
- pressure gauge support through `--pressure-gauge zero-mean`;
- intrusive FEniCS residual/Jacobian projection through
  `turtleFSI/rom/_pod_galerkin/fenics_monolithic.py`;
- reduced Newton solve utilities with line search, Jacobian reuse options, and
  least-squares correction solvers;
- a public wrapper command:

  ```text
  turtleFSI-pod-galerkin
  ```

- a one-command validation action:

  ```text
  --action validate-rollout
  ```

- linearized reduced operator bundle tools for a first offline-online branch.

## Recommended Standard Workflow

Use one wrapper command as the source of truth. Avoid hand-running the lower
level scripts unless debugging a specific stage.

### 1. Generate FOM snapshots

Run the full case with exact DVP snapshots:

```bash
turtleFSI \
  --problem TF_fsi \
  --folder simulations/pod_galerkin_training \
  --rom-export-dvp true \
  --rom-export-step 1
```

Expected output:

```text
RUN_DIR/ROM/dvp_snapshots.npz
RUN_DIR/ROM/dvp_snapshots_summary.json
```

Compute supremizers after collecting snapshots:

```bash
python -m turtleFSI.rom.compute_supremizers --run-dir RUN_DIR
```

This writes `RUN_DIR/ROM/dvp_supremizers.npz` for basis construction.

### 2. Run the standard validation rollout

Use the public wrapper for basis construction, train/test split, rollout,
metrics, and plots:

```bash
turtleFSI-pod-galerkin \
  --run-dir RUN_DIR \
  --snapshots RUN_DIR/ROM/dvp_supremizers.npz \
  --preset ballarin-validation \
  --displacement-modes 16 \
  --velocity-modes 24 \
  --supremizer-modes 12 \
  --pressure-modes 12 \
  --train-fraction 0.8 \
  --output-dir RUN_DIR/analysis
```

The preset expands to the standard validation path:

```text
--action validate-rollout
--blocked-dvp
--pressure-treatment supremizer
--pressure-dof-key fluid_pressure_dofs
--pressure-gauge zero-mean
--exact-modes
--linear-solver equilibrated-lstsq
--linear-rcond 1e-12
```

The wrapper should write:

```text
case_pod_galerkin_summary.json
basis/monolithic_pod_basis.npz
basis/monolithic_pod_basis_summary.json
validation_rollout/fenics_pod_galerkin_rollout.npz
validation_rollout/fenics_pod_galerkin_rollout_summary.json
validation_rollout/field_error_metrics.json
validation_slice_animations/
```

### 3. Run shorter debug checks when the long rollout fails

Use a one-step or short rollout to isolate projection/conditioning issues:

```bash
turtleFSI-pod-galerkin \
  --run-dir RUN_DIR \
  --snapshots RUN_DIR/ROM/dvp_supremizers.npz \
  --action step \
  --blocked-dvp \
  --pressure-treatment supremizer \
  --pressure-dof-key fluid_pressure_dofs \
  --pressure-gauge zero-mean \
  --exact-modes \
  --displacement-modes 16 \
  --velocity-modes 24 \
  --supremizer-modes 12 \
  --pressure-modes 12 \
  --target-index -1 \
  --linear-solver equilibrated-lstsq \
  --linear-rcond 1e-12
```

Then compare `initial_relative_error`, `projection_relative_error`, and
`rom_relative_error` in the step summary. If projection is good but the ROM
solve is bad, the problem is reduced dynamics/conditioning, not POD basis
representation.

## Code Changes Needed Next

### Priority 1: Make the workflow harder to misuse

- Add a named preset such as `--preset ballarin-validation` to
  `turtleFSI/rom/pod_galerkin.py`.
- The preset should imply:
  - `--action validate-rollout`;
  - `--blocked-dvp`;
  - `--pressure-treatment supremizer`;
  - `--pressure-dof-key fluid_pressure_dofs`;
  - `--linear-solver equilibrated-lstsq`;
  - validation metrics and slice plots.
- Keep all explicit CLI flags as overrides.
- Add a summary field that records every implied preset option.

Status: implemented in `turtleFSI/rom/pod_galerkin.py`. The wrapper records
`preset` and `preset_options` in `case_pod_galerkin_summary.json`.

### Priority 2: Improve mode and basis diagnostics

- Record requested mode caps, retained modes, energy threshold, and whether
  `--exact-modes` was used.
- Fail or warn when retained modes are much smaller than requested for a
  validation run.
- Report field-wise POD projection error on the held-out test window before
  running the reduced solver.

### Priority 3: Add reduced stability diagnostics

- Reduced divergence/incompressibility residual over the rollout.
- Velocity-pressure coupling rank or reduced inf-sup surrogate.
- Projected Jacobian condition number per Newton update.
- Per-field interior slice errors, not just exterior surface errors.
- Separate residual norm history for displacement, velocity, and pressure
  equation blocks where possible.

### Priority 4: Promote the Ballarin branch beyond linearized bundles

- Identify turtleFSI form terms corresponding to paper-level mass, pressure,
  convection, ALE, mesh, and structural blocks.
- Build named reduced block operators rather than anonymous projected
  Jacobians.
- Implement a reduced nonlinear time-marching solve that uses saved reduced
  operators where possible.
- Keep the full FEniCS-projected path as the reference/debug comparison.

### Priority 5: Add hyper-reduction

- Implement EIM/DEIM or another hyper-reduction method for nonaffine ALE and
  nonlinear terms.
- The acceptance target is online cost dominated by reduced dimensions and
  interpolation points, not by full residual/Jacobian assembly.

### Priority 6: Add parametric training/testing

- Store physical/geometric parameter metadata in snapshot summaries.
- Build basis training sets over multiple parameters.
- Add online tests at parameters not used in training.
- Report parameter values in every ROM summary.

## Acceptance Criteria

The code can be called a Ballarin/Rozza-style implementation when:

- exact DVP snapshots are used;
- displacement, velocity, pressure, and supremizer reduced spaces are separate
  and reported;
- pressure uses fluid pressure DoFs and a documented pressure gauge;
- supremizers are generated from the fluid incompressibility adjoint and
  embedded in the global velocity space;
- a true rollout predicts held-out snapshots without reinitializing from each
  target FOM state;
- summaries include mode counts, reconstruction errors, held-out field errors,
  reduced residual histories, conditioning diagnostics, and timings;
- the efficient online path does not assemble the full FEniCS Jacobian at every
  reduced Newton correction;
- parameterized training and independent online parameter tests are supported.

## Current Bottom Line

The code has the right intrusive projection scaffold and enough snapshot/basis
machinery to debug monolithic POD-Galerkin seriously. The main gap is not
plotting or file export anymore. The main gap is the reduced model itself:
conditioning, block nonlinear operator structure, hyper-reduction, and a
simpler preset-driven workflow that prevents accidental invalid experiments.
