# ROM Package Layout

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

## Calling a Case

For a POD/data-fit model, point the command at the snapshot bundle for the
case:

```bash
turtleFSI-pod \
  --snapshots runs/flag_pole/simulations/5s_training/1/ROM/dvp_snapshots.npz \
  --output-dir runs/flag_pole/rom/pod/5s_training
```

For a POD-Galerkin model, point the command at the simulation run directory
that contains `ROM/dvp_snapshots.npz`:

```bash
turtleFSI-pod-galerkin \
  --run-dir runs/flag_pole/simulations/5s_training/1 \
  --output-dir runs/flag_pole/rom/pod_galerkin/5s_training
```

These commands are problem-agnostic as long as the case writes the expected
snapshot arrays.

For a Ballarin/Rozza-style reduced-operator workflow, first build the blocked
basis and reduced operator bundle:

```bash
turtleFSI-pod-galerkin \
  --run-dir runs/flag_pole/simulations/5s_training/1 \
  --blocked-dvp \
  --pressure-treatment supremizer \
  --scheme ballarin-monolithic \
  --action build-operators \
  --problem TF_cfd \
  --reference-index 0 \
  --output-dir runs/flag_pole/rom/pod_galerkin/5s_training
```

Then solve the saved reduced system online:

```bash
turtleFSI-pod-galerkin \
  --run-dir runs/flag_pole/simulations/5s_training/1 \
  --blocked-dvp \
  --pressure-treatment supremizer \
  --scheme ballarin-monolithic \
  --action operator-solve \
  --operator-bundle runs/flag_pole/rom/pod_galerkin/operators/reduced_operators.npz \
  --output-dir runs/flag_pole/rom/pod_galerkin/5s_training
```

The online solve uses the saved reduced system without full FEniCS assembly.
The current bundle builder projects the monolithic residual/Jacobian linearized
at a reference snapshot. Full paper-level nonlinear/parametric EIM
hyper-reduction is still a separate extension.
