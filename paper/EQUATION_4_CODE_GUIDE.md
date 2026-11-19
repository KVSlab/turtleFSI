# Equation (4): from integrals to matrices

Reference: `paper/galerkin-paper.pdf`, section 2.5, printed page 1018.
Table I and section 2.4 supply the directional derivatives. Equation (4)
assembles those derivatives on finite-element test/trial basis functions.

## Where the matrices are formed

`turtleFSI/modules/newtonsolver.py` differentiates the complete residual with
`derivative(F, dvp_["n"], TrialFunction(DVP))` (split into linear/nonlinear
pieces at setup). `assemble(J)` forms the sparse Newton matrix. The variable
`A` there means the **entire Jacobian**, not the paper's fluid stiffness A^F.
The solve is `A * delta = -F`, followed by `state += lambda * delta`.

For example, with implicit Euler:

```text
(M^F)_ij = integral_fluid (rho_f J K / dt) phi_j . phi_i dx
(A^F)_ij = integral_fluid J K mu_f
           (grad(phi_j) G^-1 F^-1 + F^-T G^-T grad(phi_j)^T)
           F^-T G^-T : grad(phi_i) dx
```

Here trial functions supply column j and test functions supply row i. FEniCS
performs the element integration and adds contributions into a global sparse
matrix. There need not be an explicit Python loop over i and j.

`turtleFSI/modules/equation_4.py` now makes the paper's named operators and
their block sums explicit. It uses the handwritten Table-I forms in
`newton_block_forms.py` and assembles them on demand:

```python
# ns is the solver namespace AFTER fluid, solid, and extension setup,
# containing DVP, d_, v_, p_, test functions, measures, material data, etc.
from turtleFSI.modules.equation_4 import build_equation_4_forms, assemble_equation_4

forms = build_equation_4_forms(**ns)
matrices = assemble_equation_4(forms, ns.get("compiler_parameters"))
M_F = matrices["operators"]["M_F"]
A_F = matrices["operators"]["A_F"]
C_F_u = matrices["operators"]["C_F_u"]
J_dd = matrices["blocks"]["J_dd"]
J = matrices["jacobian"]
```

These are full mixed-space sparse matrices, with zeros outside each operator's
field support. They are **before** boundary-condition row replacement and
inactive-pressure diagonal repair. Apply those operations to the combined
system, not to every contribution. Reassemble after changing the Newton state
or previous timestep. Avoid `.array()` on full production meshes; dense matrices
are used only in the small regression tests.

| Code name | Paper notation / role |
|---|---|
| `M_F`, `M_S` | M^F, M^S: fluid/solid velocity time derivatives, including 1/dt |
| `M_F_d` | M^F_,d: displacement dependence of fluid inertia |
| `A_F`, `A_F_d` | Fluid viscous derivative with respect to velocity/displacement |
| `A_S` | Solid elastic force derivative with respect to displacement |
| `C_F_u`, `C_F_d` | Newton derivatives of fluid convection |
| `C_ALE_F_u`, `C_ALE_F_d` | ALE derivatives, subtracted in the momentum block |
| `B_F_p`, `B_F_d_momentum` | Pressure-force derivatives in momentum |
| `B_F_u`, `B_F_d_continuity` | Velocity/displacement derivatives in continuity |
| `F_F_d` | Fluid volume-load derivative, subtracted in momentum |
| `M_E_F`, `M_E_S` | Extension/kinematic displacement time derivatives |
| `A_E_F_d` | Extension stiffness derivative including the negative fluid-side interface flux |
| `F_E_S` | Velocity coupling in the solid kinematic equation, with a negative block sign |

E is a superscript for **extension**, not an additional unknown or an independent
matrix named E. F and S denote fluid and solid. A comma suffix denotes a
derivative. In particular, C^F_,u is not just C^F with a frozen advecting
velocity: differentiating `(grad(u) G^-1 F^-1) u` produces two terms.

## Why the code's layout looks different

The paper displays `(u^F, u^S, p^F, d^F, d^S)`. The implementation uses global
continuous displacement and velocity fields, in mixed `(d, u, p)` order.
Interface displacement/velocity degrees of freedom are shared; do not duplicate
them when comparing the paper's subdomain blocks. DVP indices can be interleaved:
use the subspace dofmaps rather than slicing equal contiguous chunks.

The code spells out:

```text
J_dd = M_E_F + A_E_F_d + M_E_S
J_du = -F_E_S
J_ud = M_F_d + A_F_d + B_F_d_momentum + C_F_d - C_ALE_F_d - F_F_d + A_S
J_uu = M_F + A_F + C_F_u - C_ALE_F_u + M_S
J_up = B_F_p
J_pd = B_F_d_continuity
J_pu = B_F_u

[ J_dd  J_du   0   ] [delta_d]     [R_extension ]
[ J_ud  J_uu  J_up ] [delta_u] = - [R_momentum  ]
[ J_pd  J_pu   0   ] [delta_p]     [R_continuity]
```

The existing component forms already include negative signs for ALE, body force,
and interface flux. The new helper normalizes the first two to the paper's
positive named operators before subtracting them. It keeps the existing positive
continuity convention described in `GALERKIN_EQUATION_AUDIT.md`.

The helper requires `theta=1`, paper extension, and a linear-elastic solid.
It does not include extra Robin supports or other problem-specific terms.
For those configurations the derivative of the full production residual remains
the authority. This is an inspection/verification path, not a replacement solver.

## What the linked POD summary means

The existing `section_3_1_smoke_100ms/basis/monolithic_pod_basis_summary.json`
describes a **basis construction** (`action: basis` in its parent summary).
It does not contain the equation-(4) matrices or demonstrate a ROM rollout.

| Entry | Meaning for this run |
|---|---|
| `num_dofs: 31965` | Length of the full mixed vector into which modes are embedded |
| `num_modes: 6` | Six reduced coefficients: two each for d, u, p |
| `coefficient_start/stop` | Python slices: d = 0:2, u = 2:4, p = 4:6 |
| `dof_key` / `num_state_dofs` | Snapshot index array and number of selected physical-field entries |
| `S_d`, `S_u`, `S_p` | Snapshot matrices: selected field DoFs by 11 training snapshots |
| `Z_d`, `Z_u_s`, `Z_p` | Basis matrices embedded in mixed space, each 31965 by 2 |
| `E_N`, `V_N`, `Q_N` | Reduced displacement, velocity, pressure spaces |
| `S_s.present: false` | No supremizer snapshots used; s here means supremizer, not solid displacement |
| `pressure_gauge: none` | The basis builder did not subtract a pressure gauge; this does not determine the FOM's boundary conditions |
| `lifting.implemented: false` | No explicit boundary lifting subtraction in this basis builder |
| `stop_index: null` | Use all available snapshots after the start index |

The selected field sizes need not sum to the embedding size: unused/unselected
mixed-vector entries remain zero in the basis. The snapshot DoF arrays determine
the selection; `num_dofs` alone does not identify active fluid pressure entries.

The basis represents `x ≈ mean + Z a`. Mean subtraction is not a general
replacement for lifting time/parameter-dependent Dirichlet data. A lifting
representation is `x ≈ lifting(t, mu) + mean_homogeneous + Z a`.

The JSON is readable metadata. `monolithic_pod_basis.npz` stores the numerical
basis, mean, singular values, and index arrays. `monolithic_pod_projection.npz`
stores training coefficients and reconstructed snapshots. NPZ is a NumPy array
archive, suitable for reloading numerical data without placing huge arrays in JSON.

Reported training reconstruction errors `||S - reconstructed_S||_F / ||S||_F`:

- displacement: 0.0083928 = **0.8393%**;
- velocity: 0.0019417 = **0.1942%**;
- pressure: 0.00016483 = **0.01648%**.

These are Euclidean coefficient errors, not FE-integral or physical-energy errors.
The aggregate mixes different physical units and is dominated by the larger
numerical scales. It is not evidence of equally accurate displacement and pressure.

The old blocked summary's `retained_energy` used the **last field's** cumulative
POD energy (pressure here). Future blocked summaries now compute the actual
combined centered-variance fraction and include `field_retained_energy` and
`metric_definitions`. The old run is preserved as historical output.

## Next replication milestones

1. Verify equation (4) against the full residual first. The new tests compare
   the named assembly with automatic differentiation and finite differences,
   check block support, and exercise identity/nonidentity shape maps, body loads,
   interface terms, and changed Newton states. Run
   `python -m pytest -q tests/test_paper_newton_forms.py` in the FEniCS environment.
2. Complete section 3.1's boundary lifting and supremizer-enriched spaces. This
   run has neither. Check the POD inner product against the paper as well: the
   current builder uses a centered Euclidean SVD. Train on a representative
   time/parameter set and reserve independent snapshots for validation.
3. Validate reduced Newton steps and free rollouts using
   `R_N(a) = Z.T R(lifting + mean + Z a)` and `J_N = Z.T J Z`, rebuilding at each
   iterate. The existing `FenicsMonolithicProblem.projected_assembly` path in
   `fenics_monolithic.py` supplies full-order residual/Jacobian projection.
   A saved linearization at a reference snapshot alone is not a nonlinear ROM.
4. Implement/validate the affine or empirical-interpolation decomposition in
   section 2.6 before claiming an online cost independent of full-order size.
   Exact projection that still assembles full FEniCS operators is useful for
   validation but does not establish that speedup.

The small algebra tests support the equation mapping; they do not establish
reproduction of the paper's full benchmarks, stability, or parameter studies.
