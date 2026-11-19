# Equation audit and corrections through Section 2.4

Reference: Ballarin and Rozza, *POD–Galerkin monolithic reduced order models for
parametrized fluid-structure interaction problems* (2016), `galerkin-paper.pdf`.

The individual fluid, linear-elastic solid, and extension derivatives were
largely correct. The corrections below address the residual/derivative mismatch,
term summation, physical interface-side selection, and Jacobian caching. They do
not replace the paper's constitutive model or introduce a different FSI model.

## Continuity and the order of inverse mappings

Section 2.1 defines the total mapping gradient as

```
H = F G = G + grad(d)
H^-1 = G^-1 F^-1
```

The implementation now uses the positive continuity residual

```
Integral_fluid J K tr(G^-1 F^-1 grad(u)) q dx.
```

This agrees with the handwritten derivatives in Table I and
`newton_block_forms.py`. Section 2.3(B) prints `F^-1 G^-1` inside the divergence;
that order conflicts with the mapping definitions and is treated here as an
apparent typographical error, not an author-confirmed erratum. The positive
continuity-row convention is retained consistently for both residual and
Jacobian; multiplying the complete constraint row by -1 gives the same equation.
With `G = I`, the ordering difference disappears.

## Equation-(3) helper totals

The dictionaries keep individual named forms and combined aliases for inspection:

- `extension_relation_f = m_e_f + a_e_f`;
- `extension_relation_s = m_e_s - f_e_s`.

The total Jacobian now excludes these aliases and includes `f_e_s` with its
negative residual sign. Previously, adding all dictionary values doubled the
fluid volume extension contribution and replaced the solid kinematic relation
with `2*m_e_s`, canceling its velocity coupling.

For compatibility, the named `f_e_s` derivative remains positive, and the combined
aliases remain available. **Do not sum all values of `equation_3_blocks` or the
component dictionaries yourself.** Use `equation_3_jacobian` for the unique,
signed sum of the paper terms. This helper does not include unrelated optional
terms such as Robin supports or viscoelastic stresses; the derivative of the full
production residual remains authoritative when those features are enabled.

## Fluid-side interface trace

Section 2.3, Remark 1 requires the extension interface flux from the fluid cell
and its outward normal. An interior-facet `+` restriction does not identify the
fluid material. Both the residual and its handwritten derivative now use a DG0
indicator, constructed from the fluid cell markers, to select the fluid trace
from `+` or `-`.

Paper extension mode requires an explicit `fsi_id`. For a nonempty interface,
`dS`, `n`, and cell markers attached to `dx_f` are required. Use `fsi_id=[]` only
when intentionally modeling no fluid-solid interface. Missing interface inputs
now raise an error instead of silently dropping a required term. Interface IDs
must identify actual fluid-solid interior facets on the same marked mesh.

## State-dependent contributions and cached Jacobians

Two fluid contributions now belong to the reassembled part of the Jacobian:

- The body-force term contains `J(d_n)`, so its derivative changes with the
  current Newton state even for a prescribed constant force.
- The `(1-theta)` mass contribution contains `J(d_(n-1))`, so its velocity
  derivative changes between timesteps.

The equations themselves are unchanged by this reassignment. It prevents those
contributions being frozen in `A_pre` at solver setup. The existing `recompute`
and `recompute_tstep` controls still govern Jacobian reuse; select both as 1 for
reassembly at every Newton iteration and timestep.

## Regression verification

`tests/test_paper_newton_forms.py` checks:

- continuity against the chain rule using `H = G + grad(d)`, for both identity
  and nontrivial shape maps;
- handwritten derivatives against automatic differentiation and centered finite
  differences;
- the complete paper helper Jacobian against the derivative of the actual
  residual, including the nonzero solid displacement/velocity coupling;
- the interface flux against an independent matrix calculation, with fluid on
  either side and different displacement gradients in neighboring cells;
- required interface inputs and explicit no-interface/non-paper configurations;
- the matrix used by production assembly after current and previous state
  changes, including a zero-force control.

Run in the configured FEniCS environment:

```sh
python -m pytest -q tests/test_paper_newton_forms.py
```

For the current Conda installation, `PKG_CONFIG_PATH` must include
`/workspace/miniforge3/envs/turtle/lib/pkgconfig` when running outside its activated
shell. These small 2D checks verify the corrected algebra and assembly; they do
not constitute a full transient, 3D, or parameter-study validation.

To reproduce the paper's unsteady model, select `LinearElastic` or
`BallarinRozzaLinearElastic`, `extrapolation="laplace"`, and
`extrapolation_sub_type="paper"`. The implicit Euler discretization introduced
in Section 2.5 corresponds to `theta=1`. The separate steady formulation in
Section 2.3, Remark 2 is outside these corrections.

### Verification results for these corrections

- All 19 cases in `tests/test_paper_newton_forms.py` passed.
- All 20 neighboring tests in `tests/test_rom_ballarin_offline_online.py` and
  `tests/test_rom_pod_galerkin.py` passed.
- A two-process MPI smoke check with `ghost_mode="shared_facet"` confirmed
  indicator volume 0.5, interface length 1, and outward x-normal +1/-1 for
  fluid on the left/right, respectively. This checks selector communication,
  not a full parallel FSI solve.

## Integration correction found by the flag run

The first `TF_fsi` paper-mode run reached production matrix assembly and exposed
an allocation error not exercised by adding separately assembled matrices in the
original regression checks. `A = Matrix(A_pre)` allocated only the cached linear
sparsity; nonlinear/interior-facet couplings could then fail with PETSc error 63
when assembling in place. `solver_setup` now initializes `A` from the complete
`J_linear + J_nonlinear` form. This allocates the union of couplings while retaining
`A_pre` for the constant contribution. The production-assembly regression now
assembles into `solver['A']` and performs the same `axpy` as the Newton loop.

The same test also exposed that `A.axpy(..., True)` incorrectly promised identical
sparsity for `A` and `A_pre`. It now uses `False`, allowing different patterns;
the cached linear matrix is only a subset of the complete Jacobian.
