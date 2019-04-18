

[![Documentation Status](https://readthedocs.org/projects/turtlefsi2/badge/?version=latest)](https://turtlefsi2.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/KVSlab/turtlefsi.svg?branch=master)](https://travis-ci.org/KVSlab/turtlefsi)

## turtleFSI - Fluid-Structure Interaction
Note: The Repository is stil under work, and we cannot guaranty any functionalty yet.

@TODO: Add gif of a "swiming" turtle.


Description
-----------
TODO: Take some text from the paper and write a two-paragraph text about turtleFSI.

The goal of turtleFSI is to provide research groups, and other individuals, with a tools ....


Authors
-------
morphMan is developed by

  * Andreas Slyngstad
  * Sebastian Gjertsen
  * Aslak W. Bergersen
  * Alban Souche 


Licence
-------
morphMan is licensed under the GNU GPL, version 3 or (at your option) any
later version.

turtleFSI is Copyright (2016-2019) by the authors.


Documentation
-------------
For an introduction to turtleFSI, and tutorials, please refer to the [documentation](https://turtlefsi2.readthedocs.io/en/latest/).

If you wish to use turtleFSI for journal publications, please cite the two master thesis's:

Slyngstad, Andreas Strøm. Verification and Validation of a Monolithic Fluid-Structure Interaction Solver in FEniCS. A comparison of mesh lifting operators. MS thesis. 2017.

Gjertsen, Sebastian. Development of a Verified and Validated Computational Framework for Fluid-Structure Interaction: Investigating Lifting Operators and Numerical Stability. MS thesis. 2017.


Installation
------------
For reference, morphMan requires the following dependencies: FEniCS > 2018.1.0, Numpy > 1.1X.
Please refer to the respective documentations for installing the dependencies on your system.
 
However, if you are on Linux or MaxOSX you can install turtleFSI through anaconda::

        conda create -n your_environment -c conda-forge turtleFSI

You can then activate your environment by runing ``source activate your_environment``.
You are now all set, and can start running fluid-structure interaction simulations.


Use
---
You can execute turtleFSI by running::
	turtleFSI --problem [path_to_problem]

turtleFSI will first look for a file locally, then check if there is one installed in turtleFSI. Please
refere to the [documentation](https://turtlefsi2.readthedocs.io/en/latest/) on how to create a problem
file and a more complete description of usage.


Contact
-------
The latest version of this software can be obtained from

  https://github.com/KVSlab/turtleFSI

Please report bugs and other issues through the issue tracker at:
  
  https://github.com/KVSlab/turtleFSI/issues
