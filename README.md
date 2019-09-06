[![Documentation Status](https://readthedocs.org/projects/turtlefsi2/badge/?version=latest)](https://turtlefsi2.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/KVSlab/turtleFSI.svg?branch=master)](https://travis-ci.org/KVSlab/turtleFSI)


# turtleFSI - a Fluid-Structure Interaction Solver

<p align="center">
    <img src="figs/turtleFSI_swim.gif" width="360" height="250" alt="turtleFSI_swim"/>
    <img src="figs/turek_benchmark.gif" width="505" height="250" alt="turtleFSI_swim"/>
</p>
<p align="center">
  left: racing turtleFSI, right: Turek benchmark FSI2
</p>


Description
-----------
turtleFSI is a monolithic fluid-structure interaction solver written in FEniCS, and has out-of-the-box high performance capabilities. The goal of turtleFSI is to provide research groups, and other individuals, with a simple, but powerfull solver to investigate fluid structure interaction problems.


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

If you wish to use turtleFSI for journal publications, please refer the two master thesis's for citation:

  * Slyngstad, Andreas Strøm. Verification and Validation of a Monolithic Fluid-Structure Interaction Solver in FEniCS. A comparison of mesh lifting operators. MS thesis. 2017.

  * Gjertsen, Sebastian. Development of a Verified and Validated Computational Framework for Fluid-Structure Interaction: Investigating Lifting Operators and Numerical Stability. MS thesis. 2017.


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
