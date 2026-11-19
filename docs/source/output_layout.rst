Output Layout and Naming
========================

Keep completed solver datasets directly under ``simulations/``. Name each
``<problem>-<requested-duration>-<material>``. For example::

    simulations/
      aneurysm-200ms-stvk/
        Checkpoint/default_variables.json
        ROM/dvp_snapshots.npz
        ROM/dvp_snapshots_summary.json
        analysis/
      flag-pole-10s-stvk/

``stvk`` means St. Venant–Kirchhoff. Keep numerical material parameters in
``Checkpoint/default_variables.json``. Duration denotes the configured target
``T``; saved snapshot times establish actual coverage.

Keep derived POD bases and validation outputs in the source dataset's
``analysis/`` directory. Do not create separate per-problem ROM collections.
Use ``results/`` for temporary experiments and manual checks. Reusable diagnostic
source belongs under ``tests/manual/``. Keep generated output ignored by Git.

Use ``folder=simulations`` with a descriptive ``sub_folder`` for new datasets.
When present, retain each visualization XDMF file with its corresponding HDF5
file. See ``simulations/README.md`` for completed datasets, available supremizers,
and the distinction between saved-data checks and paper-level validation.
