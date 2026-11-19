# Adjacency-Based Non-Intrusive ROM for `turtleFSI`

## Paper Summary

The paper proposes a **non-intrusive reduced-order model (ROM)** for fluid-structure interaction (FSI). "Non-intrusive" means the reduced model is learned from simulation data, without deriving the reduced equations directly from the full finite-element code.

The method has two stages:

1. Build a **data-driven full-order surrogate** with operators inferred from snapshots.
2. Compress that surrogate with **POD** to obtain a reduced model.

The key idea is to make operator inference practical by using **mesh adjacency**:

- Each degree of freedom (DoF) is only allowed to depend on nearby DoFs.
- Instead of solving one huge dense regression, the method solves many small local least-squares problems.
- This yields sparse inferred operators that are cheaper to store and project.

The paper splits the FSI dynamics into:

- A fluid velocity model on a reference ALE configuration.
- A solid velocity/displacement model.
- A coupling model that maps near-interface fluid states to solid forcing.
- An ALE extension model that maps solid interface displacement to fluid mesh displacement.

In approximate discrete form, the ingredients are:

- Fluid: linear term + quadratic self-interaction + bilinear ALE interaction + solid-velocity forcing + inlet forcing + constant offset.
- Solid: second-order linear dynamics in displacement/velocity, plus forcing from nearby fluid states.
- Mesh motion: linear map from solid interface displacement to fluid mesh displacement.

After those operators are inferred from data, the paper computes POD bases for:

- fluid velocity snapshots
- solid velocity snapshots

and projects the inferred sparse operators to low dimension.

The reduced model is then solved as an **implicitly coupled ROM**, because the fluid-solid coupling remains stiff even after reduction.

## What This Means for `turtleFSI`

`turtleFSI` is already an intrusive monolithic ALE FSI solver. The paper's method is not a replacement for the current solver assembly in `turtleFSI/modules/fluid.py` and `turtleFSI/modules/solid.py`. It is a new workflow that sits alongside the existing solver:

1. Run high-fidelity `turtleFSI` simulations and export training data.
2. Convert those snapshots to the reference configuration expected by the ROM method.
3. Infer sparse operators from the data using adjacency.
4. Build POD bases and project the inferred operators.
5. Add a new ROM time integrator that evolves reduced states without calling the full FEniCS Newton solve.

So the first implementation target should be:

**Use `turtleFSI` as the snapshot generator for an external/non-intrusive ROM pipeline.**

Only after that works should you consider tighter integration into the CLI.

## Important Gap Between The Paper And This Codebase

There is one major modeling mismatch:

- The paper derives the ROM around a **small-deformation approximation** and uses a **linear solid model**.
- `turtleFSI` currently solves a more general **nonlinear monolithic ALE FSI problem**, often with nonlinear solid constitutive laws and different mesh-lifting choices.

That does not prevent implementation, but it changes the scope:

- If you want to reproduce the paper closely, start from a problem close to **Hron-Turek / Turek flag**, small to moderate displacement, and keep the physics simple.
- The first ROM should target one benchmark problem, not the entire `turtleFSI` feature set.

## Recommended Implementation Phases

## Phase 1: Define the first target problem

Pick one benchmark and freeze the scope.

Recommended target:

- `turtleFSI/problems/TF_fsi.py`

Why:

- It is already the repository's main coupled benchmark.
- The tests and output conventions already exist.
- It is the closest analogue to the paper's benchmark style.

Decide up front:

- training simulation parameters
- time window to discard as transient
- snapshot interval
- quantities to reconstruct and evaluate

## Phase 2: Add a snapshot export pipeline

You need consistent training data for:

- fluid velocity
- pressure if needed for diagnostics only
- solid displacement
- solid velocity
- fluid mesh displacement / ALE displacement
- inlet/control signal
- mesh/domain/boundary metadata

In `turtleFSI`, the best integration points are:

- `initiate(...)`
- `post_solve(...)`
- possibly `save_files_visualization(...)`

Recommended implementation:

- Add a new optional output mode, for example `--rom-export True`.
- Save snapshots in a structured directory such as `results/.../ROM/`.
- Save arrays in `numpy` format (`.npy`/`.npz`) plus a JSON metadata file.

Export at each saved time:

- `u^n`: fluid velocity DoFs
- `d_s^n`: solid displacement DoFs
- `v_s^n`: solid velocity DoFs
- `d_f^n`: fluid mesh displacement DoFs
- time value
- mesh coordinates and connectivity for fluid and solid submeshes
- interface DoF mappings

For the first version, do not try to infer operators directly from mixed FEniCS objects. Export plain arrays first.

## Phase 3: Build mesh and adjacency utilities

The paper depends on local neighborhoods around each DoF.

You need utilities to derive:

- node adjacency from mesh connectivity
- DoF adjacency for vector-valued fields
- interface node/DoF sets
- "near-interface" fluid DoFs used in solid forcing
- selection matrices conceptually equivalent to `R` and `S`

Implementation tasks:

- Add a new module, e.g. `turtleFSI/rom/non_intrusive/adjacency.py`.
- Compute node adjacency from cell connectivity.
- Lift node adjacency to component-wise DoF adjacency in 2D:
  - x-velocity / y-velocity
  - x-displacement / y-displacement
- Store adjacency lists, not dense matrices.

You will also need stable indexing conventions:

- one index map for fluid DoFs
- one index map for solid DoFs
- one index map for interface DoFs

This indexing decision is foundational. Make it explicit and test it early.

## Phase 4: Map snapshots to the reference configuration

This is the most paper-specific preprocessing step.

The inferred fluid model is built on a **reference fluid mesh**, not directly on the moving mesh at each timestep.

You therefore need:

1. A reference time `t1`, ideally where interface displacement is minimal.
2. A reference fluid mesh and solid mesh.
3. A way to represent fluid snapshots on that reference mesh.

For `turtleFSI`, practical first version:

- Use the existing mesh at the chosen reference time as the reference mesh.
- Export the displacement of the fluid mesh and solid interface over time.
- Interpolate fluid velocity snapshots from deformed coordinates back to the reference coordinates.

This likely belongs in an offline script, not in the solver loop itself.

Recommended structure:

- `turtleFSI/rom/preprocess.py`
- a CLI entry or script that reads exported snapshots and writes aligned training matrices

Outputs should include:

- `U`: fluid velocity snapshot matrix
- `Ds`: solid displacement snapshot matrix
- `Vs`: solid velocity snapshot matrix
- `Df`: fluid mesh displacement snapshot matrix
- inlet/control history

## Phase 5: Implement local operator inference

This is the core of the paper.

For each target DoF `i`, solve a local regularized least-squares problem using only adjacent DoFs.

You need two inference paths:

1. Fluid subsystem inference
2. Solid subsystem inference

Fluid regression features should match the paper's approximate structure:

- constant term
- neighboring fluid velocity at current/previous step, depending on exact discretization you adopt
- local quadratic terms from velocity
- bilinear terms involving fluid velocity and mesh/solid motion
- inlet contribution

Solid regression features should include:

- neighboring solid velocity
- neighboring solid displacement
- nearby/interface fluid velocity terms
- quadratic fluid terms for interface forcing if you follow the paper closely

Recommended implementation:

- `turtleFSI/rom/non_intrusive/infer.py`
- represent each local problem with sparse index lists
- solve with `numpy.linalg.lstsq` or `scipy.linalg` / `scipy.sparse` tools

Start simpler than the paper if needed:

- first infer a linear model
- then add quadratic terms
- then add bilinear ALE/coupling terms

That staged path will make debugging much easier.

## Phase 6: Add regularization and threshold selection

The paper uses truncated SVD and selects a truncation threshold from an L-curve.

Implementation tasks:

- compute local data matrix SVDs
- evaluate several truncation thresholds `eta`
- record residual norm and coefficient norm
- choose a threshold for sampled DoFs
- interpolate those threshold values to the rest of the mesh

This can initially be simplified:

- use a fixed global truncation threshold for the first prototype
- once the pipeline works, add sampled-DoF L-curve tuning

If you skip this at the beginning, write that down explicitly. Regularization will strongly affect stability.

## Phase 7: Compute POD bases

Once the sparse full-order inferred operators exist, build reduced bases.

Needed bases:

- fluid velocity basis `Phi_f`
- solid velocity basis `Phi_s`

You may also want a basis for solid displacement, depending on how you implement the second-order solid system.

Implementation tasks:

- SVD of centered or uncentered snapshot matrices
- singular value energy plots
- basis truncation by retained energy or fixed rank

Recommended module:

- `turtleFSI/rom/common/pod.py`

Persist:

- basis vectors
- singular values
- chosen ranks

## Phase 8: Project inferred operators

With the bases in hand, project the inferred sparse operators to reduced coordinates.

This corresponds to the paper's projected operators:

- fluid reduced operators
- solid reduced operators
- coupling operators
- reduced constant/input terms

Implementation tasks:

- implement sparse-aware projection routines
- avoid building large dense Kronecker operators when possible
- define a consistent reduced-state container

Recommended module:

- `turtleFSI/rom/project.py`

For the quadratic terms, you will need a clear convention for:

- symmetric quadratic monomials
- Kronecker-product ordering
- reduced evaluation of quadratic and bilinear terms

## Phase 9: Implement the ROM time integrator

The ROM needs its own solver. It should not call the FEniCS Newton system.

Create a reduced integrator that advances:

- reduced fluid velocity
- reduced solid velocity
- reduced solid displacement

Because the coupling is stiff, the paper uses an implicit coupled solve with under-relaxation.

Implementation tasks:

- define reduced timestep update equations
- fixed-point or Newton-like implicit iteration
- convergence tolerances
- under-relaxation
- reconstruction to full field for diagnostics and plotting

Recommended module:

- `turtleFSI/rom/non_intrusive/simulate.py`

The first target is offline ROM simulation from saved reduced operators, not a drop-in replacement for `monolithic.py`.

## Phase 10: Validation against `turtleFSI` snapshots

Before integrating into the user-facing CLI, validate the ROM offline.

Compare ROM vs FOM on:

- interface displacement time histories
- lift/drag if reconstructible
- fluid velocity fields at selected times
- prediction beyond training time
- sensitivity to reduced dimensions

Mirror the paper's workflow:

- define training interval `T1`
- test on future times `t > T1`
- report average and peak errors
- sweep `(r_f, r_s)`

Recommended outputs:

- plots under `results/.../ROM_eval/`
- a JSON summary of errors and chosen hyperparameters

## Phase 11: Integrate into the CLI

Only after the offline pipeline works should you expose it in the main interface.

Possible commands:

- `turtleFSI --rom-export True ...`
- `turtleFSI --rom-train path/to/export`
- `turtleFSI --rom-run path/to/rom_model`

You may prefer a separate entry point first, for example:

- `turtleFSI-rom-export`
- `turtleFSI-rom-train`
- `turtleFSI-rom-run`

That is cleaner than overloading the existing solver entrypoint early.

## Minimal Viable Implementation Order

If the goal is to get something working quickly, do the work in this order:

1. Export fluid/solid snapshots from `TF_fsi`.
2. Build adjacency and consistent DoF indexing.
3. Preprocess snapshots onto a reference configuration.
4. Infer a **linear-only** local sparse model.
5. Build POD bases and reduced linear ROM.
6. Validate short-horizon prediction.
7. Add quadratic and bilinear terms.
8. Add sampled-DoF regularization tuning.
9. Add a polished CLI and documentation.

This order reduces risk. The full paper model is too much to debug all at once.

## Concrete Repository Changes

Recommended new package layout:

```text
turtleFSI/
  rom/
    __init__.py
    adjacency.py
    export.py
    preprocess.py
    infer.py
    pod.py
    project.py
    simulate.py
    evaluate.py
```

Recommended near-term edits:

- extend argument parsing in `turtleFSI/utils/argpar.py`
- add optional snapshot export hooks in problem `post_solve(...)` or common save utilities
- keep ROM code separate from the intrusive solver modules

## Main Technical Risks

1. **State extraction mismatch**
   `turtleFSI` stores mixed FEniCS fields; the ROM needs stable plain-array state vectors with consistent indexing.

2. **Reference-frame preprocessing**
   The ALE-to-reference mapping is essential for matching the paper's fluid model.

3. **Model-form mismatch**
   The paper uses a simplified approximate operator structure; your nonlinear `turtleFSI` data may not fit it cleanly if the regime is too nonlinear.

4. **Regularization sensitivity**
   The inferred operators can become unstable quickly without careful truncation.

5. **Quadratic term bookkeeping**
   The Kronecker/symmetric quadratic feature indexing can become error-prone.

## Recommended First Milestone

The first milestone should be:

**Generate a ROM training dataset from `TF_fsi`, infer a sparse linear adjacency-based model, and show that it predicts solid tip displacement for a short horizon better than a naive baseline.**

That milestone is small enough to finish and large enough to validate the overall approach.
