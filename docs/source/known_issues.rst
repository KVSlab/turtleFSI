.. title:: Known issues

.. _known_issues:

============
Known issues
============

MUMPS failure error message
=============
When running a large problem, typically a 3D problem with many number of degrees of freedom, with several tens of processors, 
the MUMPS solver (our default linear solver) may fail with the following error message::

 *** -------------------------------------------------------------------------
 *** Error:   Unable to solve linear system using PETSc Krylov solver.
 *** Reason:  Solution failed to converge in 0 iterations (PETSc reason DIVERGED_PC_FAILED, residual norm ||r|| = 0.000000e+00).
 *** Where:   This error was encountered inside PETScKrylovSolver.cpp.
 *** Process: 11
 *** 
 *** DOLFIN version: 2019.2.0.dev0
 *** Git changeset:  43642bad27866a5bf4e8a117c87c0f6ba777b196
 *** -------------------------------------------------------------------------

While the specific reason for this error message can vary, it may occur even when the Newton iteration appears to converge correctly.
In such a case, it is likely that the MUMPS does not hold enough memory to solve the linear system.
To provide more useful information to the user and reduce the likelihood of encountering this error message, 
we have added two lines of code to the ``newtonsolver.py`` file. 
These lines will (1) print the actual error message from the MUMPS solver and (2) allocate more memory to MUMPS. 
Specifically, the following code has been added:: 

 PETScOptions.set("mat_mumps_icntl_4", 1) 
 PETScOptions.set("mat_mumps_icntl_14", 400)

The first line of code will print the actual error message from the MUMPS solver, and the second line of code will allocate more memory to MUMPS.
For detailed information about the parameters of MUMPS, please refer to `here <https://petsc.org/release/docs/manualpages/Mat/MATSOLVERMUMPS/>`_.