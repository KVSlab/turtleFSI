.. title:: Tutorial: Create your own problem file

.. _problem_file:

======================================
Tutorial: Create your own problem file
======================================

There are a limited number of example problem files in the turtleFSI repository. Insead
we here explain how to create your own problem file.

The problem file consists of up to 7 functions:

- ``set_problem_parameters``
- ``get_mesh_domain_and_boundaries``
- ``initiate``
- ``create_bcs``
- ``pre_solve``
- ``after_solve``
- ``post_process``

each of which are called at specific points while setting up and solving the problem.


set_problem_parameters
~~~~~~~~~~~~~~~~~~~~~~
This function is meant for you to define the general parameters of the problem like, dt, end time (T),
physical parameters of the problem, etc. To see a full list of the standard parameters you can change
please refer to the ``default_variables`` defined in ``turtleFSI/problems/__init__.py``.

In ``set_problem_parameters`` you should take the ``default_variables`` as an input, and overwirte this dictionary with your own. It is particularly important to overwrite the physical variables as these vary
from problem to problem.

If you provide any command line arguments these will overwrite both the once you have defined in your problem file, and the ``default_variables``.

A simple example of this function can look like this::

        def set_problem_parameters(default_variables, **namespace):
            default_variables.update(dict(
                # Temporal variables
                T = 30,                    # End time [s]
                dt = 0.01,                 # Time step [s]
                theta = 0.51,              # Temporal scheme

                # Physical constants ('FSI 3')
                Um = 2.0,                  # Max. velocity inlet, CDF3: 2.0 [m/s]
                rho_f = 1.0e3,             # Fluid density [kg/m3]
                mu_f = 1.0,                # Fluid dynamic viscosity [Pa.s]
                rho_s = 1.0e3,             # Solid density[kg/m3]
                nu_s = 0.4,                # Solid Poisson ratio [-]
                mu_s = 2.0e6,              # Shear modulus, CSM3: 0.5E6 [Pa]
                lambda_s = 4e6,            # Solid Young's modulus [Pa]

                # Problem specific
                folder = "simple_results") # Name of the results folder


get_mesh_domain_and_boundaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This is the only function which you at the very minimum have to provide. Here you read in or define your
mesh, domain markers, and boundary markers. In ``turtle_demo.py`` there is an example of reading in
all the variables, while in ``TF_fsi.py`` the domain and boundaries are marked using FEniCS functions.

If you have any questions regarding how to best create a mesh please refer to the FEniCS documentation, but
``pygmsh`` in combination with ``meshio`` create a lot of geometries.

A short example using built in FEniCS functions could look like this::

        def get_mesh_domain_and_boundaries(**namespace):
            TODO: For instance a backward facing step?

initiate
~~~~~~~~
TODO

create_bcs
~~~~~~~~~~
TODO

pre_solve
~~~~~~~~~
TODO

after_solve
~~~~~~~~~~~
TODO

post_process
~~~~~~~~~~~~
TODO
