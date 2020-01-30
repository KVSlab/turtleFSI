.. title:: Solver verification and performance

.. _verif_perf:

==================
Solver performance
==================

turtleFSI benefits from the high performance computing (HPC) functionability
of FEniCS and the solver can be executed with MPI parallel tasks as follow::

 mpirun -np 4 turtleFSI

We performed a strong scaling of a 3D FSI pipe flow (Figure a-b) to illustrate the behaviour of the
the solver using a relatively large number of parallel MPI tasks. The simulation consists at imposing
a constant Dirichlet plug flow at one end of the fluid domain and compute the resulting structure
deformation and fluid flow along the pipe. We present the results obtained at the second time step
of the simulation starting from initial rest.
We demonstrate an adequate scaling up to more than 100 cores, both executing
turtleFSI from a module installation or within a docker container (Figure c).
A direct consequence of splitting the geometry in several MPI domains (Figure b) is an increase of
the system size associated with the handling of the degree of freedoms along the
inner split boundaries. We illustrate this effect in Figure d where the total memory usage
is monitored as function of the number of MPI tasks used to solve the problem.


.. figure:: ../../figs/figure_HPC.png
    :width: 600px
    :align: center

    Figure: Strong scaling of a 3D FSI pipe flow problem. a) Meshing of the inner fluid domain,
    and outer solid pipe with a total of ca. 63000 elements. b) Split of the geometry in
    16 MPI domains. c) Time of computing and assembling the Jacobian matrix once (jac(1)) and
    solving 5 Newton's iterations (it(5)) over one time step,
    as function of the number of MPI tasks. Notice that most of the compute time is spent
    on the Jacobian operations. Results obtained with a mesh discretization of 63000 elements.
    d) System memory usage as function of the number of MPI tasks for 3 different mesh discretizations
    of the problem illustrated in a).
