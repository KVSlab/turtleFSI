.. title:: Installation

.. _installation:

============
Installation
============

Compatibility and Dependencies
==============================
The dependencies of turtleFSI are:

* FEniCS 2019.1.0
* Numpy >1.1X
* Python >=3.7

Basic Installation
==================
If you have a MacOX or Linux operating system we recommend that you
install turtleFSI through Anaconda. First, install `Anaconda or Miniconda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html#anaconda-or-miniconda>`_,
depending on your need. For just installing turtleFSI we recommend Miniconda.
Then execute the following command in a terminal window::

    $ conda create -n your_environment -c conda-forge turtleFSI

You can then activate your environment by running ``source activate your_environment``.
Now you are all set, and can start using turtleFSI. A detailed explanation for usage of
turtleFSI can be found `here <https://turtlefsi2.readthedocs.io/en/latest/using_turtleFSI.html>`_.

If you are using turtleFSI on a high performance computing (HPC) cluster we always
recommend that you build from source, as described below. This is in accordance
with the guidelines provided by the `FEniCS project <https://fenicsproject.org/download/>`_
users to install FEniCS from source when on a HPC cluster. 

Development version
===================

Downloading
~~~~~~~~~~~
The latest development version of turtleFSI can be found on the official
`turtleFSI git repository <https://github.com/KVSlab/turtleFSI>`_ on Github.
To clone the turtleFSI repository, open a terminal, navigate to the directory where you wish
turtleFSI to be stored, type the following command, and press Enter::

    $ git clone https://github.com/KVSlab/turtleFSI

After the source distribution has been downloaded, all the files will be located
in the newly created ``turtleFSI`` folder.

Building
~~~~~~~~
In order to build and install turtleFSI, navigate into the ``turtleFSI`` folder, where a ``setup.py``
file will be located. First, make sure that all dependencies are installed.
Then, you can install turtleFSI be executing the following::

    $ python setup.py install

If you are installing turtleFSI somewhere you do not have root access, typically on a cluster, you can add
``--user`` to install locally.