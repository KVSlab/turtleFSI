# turtleFSI
Monolithic Fluid-Structure Interaction (FSI) solver

Runs with FEniCS 2017.2 (conda install and docker quay.io/fenicsproject/stable:2017.2.0):

Examples of use: 

python3 monolithic.py -problem=TF_fsi

python3 monolithic.py -problem=TF_cfd

python3 monolithic.py -problem=TF_csm -extravar=biharmonic -solidvar=csm

python3 monolithic.py -problem=turtle_demo -extravar=biharmonic -theta=0.5025 -solver=reusejac

