.. title:: Installation

.. _installation:

============
Installation
============

Compatibility and Dependencies
==============================
The general dependencies of turtleFSI are:

* FEniCS 2019.1.0
* Numpy >1.1X
* Python >=3.5

Basic Installation
==================
If you have a MacOX or Linux operating system we recommend that you
install turtleFSI through Anaconda. First, install Anaconda or Miniconda.
Then execute the following command in a terminal window::

    $ conda create -n your_environment -c conda-forge turtleFSI

You can then activate your environment by running ``source activate your_environment``.
Now you are all set, and can start using turtleFSI. You can execute turtleFSI directly
through the terminal by typing::

    $ turtleFSI

followed by any additional options, for instance, which problem to run and the time step size.
Use ``-h`` to see all available options. A detailed explanation for usage of turtleFSI can be found
here_.

If you are using turtleFSI on a high performance computing (HPC) cluster we always recommend that you
build from source. In the 2019.1 version of FEniCS the conda version does not have HDF5 support
in parallel. The FEniCS project also recommends_ users to install FEniCS from source when on a
HPC cluster.

.. _here: <https://turtlefsi2.readthedocs.io/en/latest/using_turtleFSI.html>.
.. _recommends: <https:www/test.no>

Development version
===================

Downloading
~~~~~~~~~~~
The latest development version of turtleFSI can be found on the official
`turtleFSI git repository <https://github.com/KVSlab/turtleFSI>`_ on Github.
To clone the turtleFSI repository, open a terminal, navigate to the directory where you wish
turtleFSI to be stored, type the following command, and press Enter::

    $ git clone https://github.com/KVSlab/turtleFSI

After the source distribution has been downloaded, all the files required will be located
in the newly created ``turtleFSI`` folder.

Building
~~~~~~~~
In order to build and install turtleFSI, navigate into the ``turtleFSI`` folder, where a ``setup.py``
file will be located. First, make sure that all dependencies are installed.
Then, you can install turtleFSI be executing the following::

    $ python setup.py install


If you are on a cluster or server where you do not have root access, you can add ``--user`` to install
turtleFSI locally.
