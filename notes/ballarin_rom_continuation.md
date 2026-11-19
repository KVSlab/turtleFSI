# Ballarin POD-Galerkin ROM Continuation Notes

This branch is for implementing the Ballarin/Rozza POD-Galerkin monolithic FSI ROM.

## Current Branch

Use this branch for the ROM work:

```bash
git switch ballarin-table-i-components
```

Your fork remote is expected to be:

```bash
personal  https://github.com/marissal19/turtleFSI.git
```

Check with:

```bash
git remote -v
```

## Important Files

Section 2.3 weak forms:

```text
turtleFSI/modules/common.py
turtleFSI/modules/fluid.py
turtleFSI/modules/solid.py
turtleFSI/modules/laplace.py
```

Table I / Equation 3 component forms:

```text
turtleFSI/modules/newton_block_forms.py
```

FEniCS projection and Equation 3 block summation/check:

```text
turtleFSI/rom/_pod_galerkin/fenics_monolithic.py
```

Command-line checker for projected residual/Jacobian and Equation 3 comparison:

```text
turtleFSI/rom/_pod_galerkin/check_fenics_pod_galerkin_projection.py
```

ROM operator bundle path that should be updated next:

```text
turtleFSI/rom/_pod_galerkin/build_ballarin_operator_bundle.py
```

## What Is Implemented So Far

- Ballarin/Rozza Section 2.3-style residual terms are represented in the fluid, solid, and extension modules.
- Table I / Equation 3 derivative component forms are exposed in `newton_block_forms.py`.
- `FEniCSMonolithicProblem.projected_equation_3_component_blocks(...)` assembles and projects named Equation 3 components.
- `FEniCSMonolithicProblem.check_equation_3_projection(...)` compares the summed Equation 3 component operator against the full projected Jacobian `Phi.T J Phi`.
- `check_fenics_pod_galerkin_projection.py --check-equation-3` writes the comparison matrices and summary diagnostics.
- `build_ballarin_operator_bundle.py` now saves/uses the component-assembled Equation 3/4 reduced operator blocks and keeps the aggregate projected Jacobian only as a diagnostic.
- Blocked DVP basis construction builds separate global displacement, global velocity, and fluid-pressure bases; supremizer modes, when requested, are embedded into the global velocity space.

## Next Technical Step

Run the Equation 3 check in an environment with FEniCS/dolfin available:

```bash
python -m turtleFSI.rom._pod_galerkin.check_fenics_pod_galerkin_projection \
  --problem TF_fsi \
  --basis path/to/monolithic_pod_basis.npz \
  --snapshots path/to/dvp_snapshots.npz \
  --output-dir runs/eq3_check \
  --theta 1.0 \
  --check-equation-3
```

Inspect:

```text
runs/eq3_check/fenics_projected_assembly_summary.json
```

The key fields are:

```text
equation_3_difference_norm
equation_3_relative_difference_norm
equation_3_component_count
```

If `equation_3_relative_difference_norm` is near machine precision, the component-summed Equation 3 operator matches the full projected Jacobian.

The previous next step, updating `build_ballarin_operator_bundle.py` to save/use the Equation 3 component-assembled operator instead of only the aggregate `J_ref` projection, is complete. The remaining near-term work is to run this check in a FEniCS environment, then use the diagnostics to improve the nonlinear rollout and reduced stability reporting.

## Working Locally On Another Computer

Clone your fork or add your fork as a remote in an existing checkout:

```bash
git remote add personal https://github.com/marissal19/turtleFSI.git
```

Fetch and switch to the branch:

```bash
git fetch personal
git switch ballarin-table-i-components
```

If the branch does not exist locally yet:

```bash
git switch -c ballarin-table-i-components --track personal/ballarin-table-i-components
```

Before editing locally, update from GitHub:

```bash
git pull --ff-only
```

Make changes locally, then commit and push:

```bash
git status --short
git add path/to/changed/files
git commit -m "Describe the ROM change"
git push
```

## Syncing Changes Back To This Computer

When you return to this workspace, run:

```bash
cd /workspace/turtleFSI
git switch ballarin-table-i-components
git fetch personal
git pull --ff-only
```

That will bring in changes you pushed from your local computer.

If `git pull --ff-only` fails, it means this workspace has local commits or uncommitted edits that conflict with GitHub. Check:

```bash
git status --short
git log --oneline --decorate -5
```

Do not run `git reset --hard` unless you are sure you want to discard local work.

## Helpful Verification Commands

Run available non-FEniCS tests:

```bash
.venv/bin/python -m pytest tests/test_rom_ballarin_offline_online.py -q
```

Compile key files:

```bash
.venv/bin/python -m py_compile \
  turtleFSI/modules/newton_block_forms.py \
  turtleFSI/rom/_pod_galerkin/fenics_monolithic.py \
  turtleFSI/rom/_pod_galerkin/check_fenics_pod_galerkin_projection.py
```

Check whether FEniCS is available:

```bash
python - <<'PYTHON_CHECK'
try:
    import dolfin
    print("dolfin available", dolfin.__version__)
except Exception as exc:
    print("dolfin unavailable:", type(exc).__name__, exc)
PYTHON_CHECK
```
