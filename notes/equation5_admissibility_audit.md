# Equation (5): auxiliary velocity-space admissibility audit

Reference: `paper/galerkin-paper.pdf`, Section 3.1, equation (5), and the lifting
footnote on p. 1021. The paper specifies a velocity-space Riesz map and global
supremizer snapshots, with displacement/velocity snapshots adjusted for
inhomogeneous boundary data. For this flag problem, homogeneous velocity
increments vanish on markers 2, 3, 6 and 7. They must remain free on the moving
fluid-solid interface (5) and velocity outlet (4).

## Implementation before the correction

- `rom/compute_supremizers.py`: solves global H1 Riesz problem without essential
  velocity boundary conditions. The right-hand side uses fluid incompressibility
  with the saved deformation and geometry map.
- `rom/_pod_galerkin/build_monolithic_pod_basis.py`: centers supremizer snapshots,
  computes a POD, and orthogonalizes enrichment against the velocity POD.
- `rom/_pod_galerkin/fenics_monolithic.py`: zeros constrained rows of the resulting
  increment basis at rollout time. This ensures boundary values but does not
  preserve the auxiliary constrained Riesz equation.

The paper's space notation does not spell out this benchmark's boundary markers.
The homogeneous constraints above follow from the actual FOM test/increment
space after accounting for its inlet lifting; this is the benchmark-specific
interpretation of admissibility, not a literal marker prescription in the paper.

## Measured verification

At R00L00, saved FOM state t=10.01 s, the recomputed unconstrained supremizer
matches the cached training snapshot exactly. The global collapsed velocity
space has 10,216 DoFs, of which 618 are constrained velocity DoFs. The diagnostic
checks the boundary-derived collapsed DoF set against the actual mixed FOM
constraints.

| Auxiliary vector | Homogeneous velocity BCs | Free-row relative Eq. (5) residual |
| --- | --- | --- |
| Current unconstrained solve | violated | 1.82e-13 |
| Same solution with boundary entries zeroed | satisfied | 3.19e2 |
| Riesz solve with homogeneous BCs imposed during solve | satisfied | 1.43e-14 |

The clipped and constrained solutions differ by 40.03 in relative global H1
norm. For the completed 40-mode ROM's five raw supremizer modes, 8.5–12.7% of
each mode's squared coefficient norm is on the constrained velocity DoFs before
the rollout zeros them. This is a coefficient diagnostic, not physical energy.

The solve-then-zero operation demonstrably fails to produce the constrained
Riesz representative. This is a concrete inconsistency to address; it is not
proof that fixing it alone resolves every pressure error or the 56-mode drift.
The audit itself did not change existing snapshots.

## Experimental correction (reverted)

A constrained auxiliary solve and admissible-space POD orthogonalization were
implemented and tested. The user-requested 40-mode rerun completed but pressure
error increased from approximately 405% to 2670%. At the user's request, these
production changes and their regression tests were reverted. The original
unconstrained equation-(5) solve and original POD orthogonalization are active
again. Existing experiment outputs and this diagnostic history are preserved.
The result does not establish whether all boundary treatments are equivalent;
it shows that the experimental correction did not improve this reduced model.

## Reproduce without training or rollout

```bash
python -m turtleFSI.cases.audit_supremizer_admissibility \
  --run-dir simulations/aero/flag-training-completed/results/radius_00_length_00 \
  --basis rom-results/aero/flag-training-completed/analysis/baseline-mode-study-20261116-153122-303661248/d3-v20-p12-s5/basis/monolithic_pod_basis.npz \
  --output-dir /path/to/new/admissibility-audit --time 10.01
```

This performs two auxiliary linear solves only and writes `admissibility.json`.
It leaves production supremizer generation, POD training and rollout unchanged.
