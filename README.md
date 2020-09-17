[![Documentation Status](https://readthedocs.org/projects/turtlefsi2/badge/?version=latest)](https://turtlefsi2.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/KVSlab/turtleFSI.svg?branch=master)](https://travis-ci.org/KVSlab/turtleFSI)
[![status](https://joss.theoj.org/papers/b7febdaa2709205d40b51227091c3b0b/status.svg)](https://joss.theoj.org/papers/b7febdaa2709205d40b51227091c3b0b)

# turtleFSI - a Fluid-Structure Interaction Solver

<p align="center">
    <img src="figs/turtleFSI_swim.gif" width="288" height="200" alt="turtleFSI_swim"/>
    <img src="figs/turek_benchmark.gif" width="404" height="200" alt="turtleFSI_swim"/>
</p>
<p align="center">
  To the left we show a turtle swimming (in turtleFSI), and to the right, the classical Turek benchmark (FSI2).
</p>


Description
-----------
turtleFSI is a monolithic fluid-structure interaction solver written in FEniCS, and has out-of-the-box high performance capabilities. The goal of turtleFSI is to provide research groups, and other individuals, with a simple, but robust solver to investigate fluid structure interaction problems.


Authors
-------
turtleFSI is developed by:

  * Andreas Slyngstad
  * Sebastian Gjertsen
  * Aslak W. Bergersen
  * Alban Souche
  * Kristian Valen-Sendstad


Licence
-------
turtleFSI is licensed under the GNU GPL, version 3 or (at your option) any
later version. turtleFSI is Copyright (2016-2019) by the authors.


Documentation
-------------
For an introduction to turtleFSI, and tutorials, please refer to the [documentation](https://turtlefsi2.readthedocs.io/en/latest/).

If you wish to use turtleFSI for journal publications, please refer to the [JOSS publication](https://joss.theoj.org/papers/10.21105/joss.02089#):

Bergersen et al., (2020). turtleFSI: A Robust and Monolithic FEniCS-based Fluid-Structure Interaction Solver. Journal of Open Source Software, 5(50), 2089, https://doi.org/10.21105/joss.02089

Installation
------------
turtleFSI is build upon the open source Finite Elements FEniCS project (version 2018.1.0 or 2019.1.0).
Please refer to the respective FEniCS documentation for installing the dependencies on your system.

However, if you are using Linux or MaxOSX you can install turtleFSI through anaconda::

        conda create -n your_environment -c conda-forge turtleFSI

You can then activate your environment by runing ``source activate your_environment``.
You are now all set, and can start running fluid-structure interaction simulations.


Use
---
Run turtleFSI with all the default parameters::
   ``turtleFSI``

See all the command line parameters run the following command::
  ``turtleFSI -h``

Run a specific problem file::
  ``turtleFSI --problem [path_to_problem]``

When calling a specific problem file, turtleFSI will first look for the file name locally, then check if the file name is present in the directory "/turtleFSI/problems/".
Please refere to the [documentation](https://turtlefsi2.readthedocs.io/en/latest/) to learn how to define a new problem file and for a more complete description of usage.


Contact
-------
The latest version of this software can be obtained from

  https://github.com/KVSlab/turtleFSI

Please report bugs and other issues through the issue tracker at:

  https://github.com/KVSlab/turtleFSI/issues
