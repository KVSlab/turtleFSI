# Manual integration checks

`flag_paper_check.py` exercises TF_fsi geometry with the paper's linear-elastic
solid, paper Laplace extension, and implicit Euler time stepping. It checks the
assembled Jacobian, residual, finite state, and sampled deformation Jacobian.
It is diagnostic source code, not a completed simulation or a training dataset.

From the repository root, in the configured FEniCS environment:

```bash
PYTHONPATH="$PWD/tests/manual:$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  turtleFSI -p flag_paper_check
```

Outputs go to `results/manual/flag-paper-check/`, outside the retained datasets.
A successful short check does not establish agreement with published benchmarks.
