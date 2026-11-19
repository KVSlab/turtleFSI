POD-Galerkin ROM Implementation Plan
====================================

Source Paper
------------

The implementation target is the local paper:

``paper/galerkin-paper.pdf``
    Ballarin and Rozza, "POD-Galerkin monolithic reduced order models for
    parametrized fluid-structure interaction problems", International Journal
    for Numerical Methods in Fluids, 2016.

This is a monolithic POD-Galerkin FSI paper. It is not the later
semi-implicit split FSI paper.

What the Paper Requires
-----------------------

The paper's ROM has these essential ingredients:

1. A monolithic high-fidelity FSI formulation.

2. Separate reduced spaces for:

   * global displacement,
   * velocity,
   * pressure,
   * velocity supremizers.

3. Supremizer enrichment of the reduced velocity space for pressure stability.

4. Galerkin projection of the monolithic algebraic FSI system over the reduced
   displacement, velocity, and pressure spaces.

5. Offline-online decomposition for efficient repeated online solves.

6. Empirical interpolation or equivalent hyper-reduction for nonlinear and
   parametrized tensors when direct online assembly would still depend on the
   full-order dimension.

Current turtleFSI Status
------------------------

``turtleFSI/rom/pod.py``
    A baseline data-fit POD model. It builds POD bases and learns a reduced
    linear update. This is not the Ballarin/Rozza POD-Galerkin method.

``turtleFSI/rom/pod_galerkin.py``
    The correct public entry point for the physics ROM. It can build a
    monolithic POD basis from exact ``dvp`` snapshots and can project the
    turtleFSI residual/Jacobian for reduced Newton solves.

``turtleFSI/rom/_pod_galerkin/build_monolithic_pod_basis.py``
    Already supports a useful ``--blocked-dvp`` mode that builds separate
    displacement, velocity, and pressure bases from exact DVP snapshots.

``turtleFSI/rom/_pod_galerkin/fenics_monolithic.py``
    Rebuilds the turtleFSI monolithic FEniCS problem and exposes residual and
    Jacobian projection hooks.

What Is Still Missing
---------------------

The current code is closer to the correct paper than a data-fit ROM, but it is
not an exact Ballarin/Rozza implementation yet.

Missing items:

1. FEniCS extraction of the specific Ballarin/Rozza monolithic operator blocks
   from turtleFSI forms.

2. Automated construction of saved reduced operator bundles from the turtleFSI
   FEniCS forms.

3. EIM or another hyper-reduction method for nonlinear and parametrized terms.

4. Parameter handling for physical and geometric parameters.

Implemented pieces:

* blocked displacement, velocity, and pressure POD spaces;
* offline supremizer computation through
  ``python -m turtleFSI.rom.compute_supremizers --run-dir RUN``;
* ROM-side supremizer POD basis construction when supremizer snapshots are
  already present in the snapshot bundle;
* velocity-space enrichment by appending supremizer modes to the mixed basis;
* block coefficient output for displacement, velocity, supremizer, and pressure
  modes.
* block state helpers in ``ballarin_monolithic.py`` for ``d_N``, ``u_N``,
  ``s_N``, and ``p_N``;
* offline projection utilities in ``offline_online.py`` for reduced linear
  operators and quadratic tensors.
* saved reduced operator bundles and a reduced-only online solve path through
  ``--scheme ballarin-monolithic --action operator-solve``.

The block reduced unknown layout is now:

   ``d_N``
       reduced displacement coefficients.

   ``u_N``
       reduced velocity coefficients.

   ``s_N``
       reduced supremizer enrichment coefficients.

   ``p_N``
       reduced pressure coefficients.

Implementation Phases
---------------------

Phase 1: Correctly Label Current Capabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Do not call the current solver "exact Ballarin". The public command now keeps
the default as:

.. code-block:: bash

   turtleFSI-pod-galerkin --scheme monolithic

The exact paper target is reserved as:

.. code-block:: bash

   turtleFSI-pod-galerkin --scheme ballarin-monolithic

Until all missing pieces are implemented, that mode must raise a clear
``NotImplementedError``.

Phase 2: Blocked DVP Bases
~~~~~~~~~~~~~~~~~~~~~~~~~~

Blocked bases are implemented through ``--blocked-dvp``. This is the current
path for projecting separate displacement, velocity, and pressure reduced
spaces while preserving the full mixed DVP vector layout:

.. code-block:: bash

   turtleFSI-pod-galerkin \
     --run-dir simulations/5s_training/1 \
     --blocked-dvp \
     --displacement-modes 12 \
     --velocity-modes 20 \
     --pressure-modes 10

Required behavior:

* build separate POD spaces for displacement, velocity, and pressure;
* preserve the mixed FEniCS DVP vector layout;
* report reconstruction error per field;
* save reduced coefficient histories by field block;
* make ``Phi.T R`` equivalent to separate block projections against the
  displacement, velocity, and pressure basis columns;
* keep interface continuity inherited from global displacement/velocity modes.

Remaining Ballarin/Rozza work after this phase:

* map turtleFSI/FEniCS forms into named Ballarin/Rozza operator blocks;
* automatically build preassembled reduced operators from turtleFSI/FEniCS
  forms.

Phase 3: Supremizer Enrichment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ballarin/Rozza solve an auxiliary supremizer problem during the offline stage
to stabilize pressure. In implementation terms, this means:

1. For each pressure snapshot, solve a supremizer problem. This is implemented
   after DVP export with:

   .. code-block:: bash

      python -m turtleFSI.rom.compute_supremizers --run-dir RUN

2. Store supremizer snapshots in ``ROM/dvp_snapshots.npz`` or a companion
   bundle. The offline command writes ``supremizer_snapshots`` to
   ``ROM/dvp_supremizers.npz``. The ROM basis
   builder reads any of these arrays:
   ``supremizer``, ``supremizers``, ``supremizer_snapshots``,
   ``velocity_supremizer``, or ``velocity_supremizer_snapshots``.

3. Build a supremizer POD basis. This is implemented when the snapshots exist.

4. Enrich the velocity basis. This is implemented by appending supremizer modes
   on the velocity DOFs:

   ``V_N = span(velocity_modes, supremizer_modes)``

5. Add CLI support:

   .. code-block:: bash

      turtleFSI-pod-galerkin \
        --pressure-treatment supremizer

6. Add tests that pressure mode instability is detected when supremizers are
   absent. This diagnostic is still missing.

Phase 4: Block Algebraic Projection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The exact paper does not just project one anonymous ``dvp`` vector. It writes
the reduced approximation as block variables:

``d ≈ Z_d d_N``
    displacement expansion.

``u ≈ Z_u,s u_N``
    velocity plus supremizer-enriched expansion.

``p ≈ Z_p p_N``
    pressure expansion.

Implementation tasks:

1. Add a block reduced basis object, for example:

   ``turtleFSI/rom/_pod_galerkin/block_basis.py``

2. Add a block reduced state object with ``d``, ``u``, and ``p`` coefficient
   arrays.

3. Project the monolithic residual by block:

   * displacement equations against ``Z_d``;
   * velocity equations against ``Z_u,s``;
   * pressure equations against ``Z_p``.

4. Solve the reduced block Newton system.

5. Save reduced coefficient histories by block.

Phase 5: Offline-Online Efficiency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current projected FEniCS path still touches full-order assembly during the
online solve. That is useful for correctness, but it is not the efficient
offline-online procedure emphasized in the paper. The first code layer for this
now exists in ``offline_online.py``: it projects full linear operators and
quadratic tensors into reduced coordinates. A saved reduced operator bundle can
now be solved online without full FEniCS assembly.

Required work:

1. Identify the monolithic turtleFSI forms corresponding to the paper's mass,
   stiffness, convection, pressure, mesh/ALE, and structure terms.

2. Precompute reduced linear operators where possible. Generic projection
   utilities are implemented; the remaining work is extracting the actual
   turtleFSI operator blocks.

3. For nonlinear terms, implement empirical interpolation or another
   hyper-reduction strategy. Dense quadratic tensor projection is available for
   small/reference tests, but not yet a scalable FEniCS EIM path.

4. Ensure online cost scales mostly with the reduced dimension, not the
   full-order dimension.

5. Keep the current FEniCS-projected path as a reference implementation for
   validating the hyper-reduced operators.

Phase 6: Parametric Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The paper is parametrized. To match it more closely:

1. Store parameter values with each training snapshot bundle.

2. Build training sets over parameters.

3. Support new online parameter values.

4. Add affine/EIM coefficient evaluation for parameter-dependent operators.

Expected File Changes
---------------------

Modify:

``turtleFSI/rom/pod_galerkin.py``
    Public workflow, scheme selection, validation, summaries.

``turtleFSI/rom/_pod_galerkin/build_monolithic_pod_basis.py``
    Promote blocked DVP basis construction and add supremizer basis support.

``turtleFSI/rom/_pod_galerkin/monolithic_pod.py``
    Add block basis structures or delegate them to a new module.

``turtleFSI/rom/_pod_galerkin/fenics_monolithic.py``
    Expose block residual/Jacobian projection and auxiliary supremizer solves.

``turtleFSI/cases/snapshots/dvp_export.py``
    Export stable DVP block metadata, parameter metadata, and optional
    supremizer snapshots.

Add:

``turtleFSI/rom/_pod_galerkin/block_basis.py``
    Displacement/velocity/pressure/supremizer basis definitions.

``turtleFSI/rom/_pod_galerkin/supremizer.py``
    Supremizer solve, snapshot collection, and POD basis construction.

``turtleFSI/rom/_pod_galerkin/ballarin_monolithic.py``
    Paper-style block reduced algebraic system.

``turtleFSI/rom/_pod_galerkin/offline_online.py``
    Reduced operator assembly and online coefficient evaluation.

``turtleFSI/rom/_pod_galerkin/build_ballarin_operator_bundle.py``
    Builds a saved reduced operator bundle from turtleFSI/FEniCS residual and
    Jacobian forms linearized at a reference snapshot.

``turtleFSI/rom/_pod_galerkin/eim.py``
    Empirical interpolation/hyper-reduction utilities.

Acceptance Criteria
-------------------

The implementation can be called "Ballarin/Rozza 2016 style" only when:

* ``--scheme ballarin-monolithic`` runs ``build-operators`` and
  ``operator-solve`` without ``NotImplementedError``;
* separate displacement, velocity, pressure, and supremizer spaces are used;
* pressure stabilization uses supremizer enrichment;
* the linearized reduced equations are block Galerkin projections of the
  monolithic FSI residual/Jacobian;
* online solves do not require rebuilding the full FEniCS Jacobian at every
  reduced Newton correction except in the reference/debug path;
* summaries report basis sizes for every block, pressure treatment, reduced
  residual norms, Newton convergence, and online timing;
* the workflow works for each compatible case folder under ``runs/<case>``.

Remaining Paper-Level Gap
-------------------------

The current Ballarin/Rozza-style path is an executable offline-online
linearized POD-Galerkin workflow. Matching the full nonlinear parametrized
paper still requires empirical interpolation or another hyper-reduction method
for nonlinear and parameter-dependent operators, plus parameter metadata in the
training bundles.
