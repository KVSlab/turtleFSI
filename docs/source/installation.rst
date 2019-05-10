.. title:: Installation

.. _installation:

============
Installation
============
turtleFSI is a entry-level simple FSI solver. The solver is slow, but a perfect starting point for exploring problems with FSI. The project is accessible through
`GitHub <https://github.com/KVSlab/turtleFSI/>`_.

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
Now you are all set, and can start using turtleFSI.
You can run turtleFSI directly through the terminal, by typing::

    $ turtleFSI

followed by the command line arguments. Use ``-h`` to see all the options.
A detailed explanation for usage of turtleFSI is described in :ref:`getting_started`.


Development version
===================

Downloading
~~~~~~~~~~~
The latest development version of turtleFSI can be found on the official
`turtleFSI git repository <https://github.com/KVSlab/turtleFSI>`_ on Github.
Make sure Git (>=1.6) is installed, which is needed to clone the repository.
To clone the turtleFSI repository, navigate to the directory where you wish
turtleFSI to be stored, type the following command, and press Enter::

    $ git clone https://github.com/KVSlab/turtleFSI

After the source distribution has been downloaded, all the files required will be located
in the newly created ``turtleFSI`` folder.

Building
~~~~~~~~
In order to build and install turtleFSI, navigate into the ``turtleFSI`` folder, where a ``setup.py``
file will be located. First, make sure that all dependencies are installed. Then, building and installation of turtleFSI
can be performed by simply running the following command from the terminal window::

    $ python setup.py install
