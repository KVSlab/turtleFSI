from dolfin import *
from turtleFSI.modules import common
import numpy as np
import ufl  # ufl module
from os import path
#from turtleFSI.problems import *

def project_solid(tensorForm, fxnSpace, dx_s):
    #
    # This function projects a UFL tensor equation (tensorForm) using a tensor function space (fxnSpace)
    # on only the solid part of the mesh, given by the differential operator for the solid domain (dx_s)
    #
    # This is basically the same as the inner workings of the built-in "project()" function, but it
    # allows us to calculate on a specific domain rather than the whole mesh
    #
    v = TestFunction(fxnSpace) 
    u = TrialFunction(fxnSpace)
    a=inner(u,v)*dx_s # bilinear form
    L=inner(tensorForm,v)*dx_s # linear form
    tensorProjected=Function(fxnSpace) # output tensor-valued function
     
    # Alternate way that doesnt work on MPI (may be faster on PC)
    #quadDeg = 4 # Need to set quadrature degree for integration, otherwise defaults to many points and is very slow
    #solve(a==L, tensorProjected,form_compiler_parameters = {"quadrature_degree": quadDeg}) 
 
    '''
    From "Numerical Tours of Continuum Mechanics using FEniCS", the stresses can be computed using a LocalSolver 
    Since the stress function space is a DG space, element-wise projection is efficient
    '''
    solver = LocalSolver(a, L)
    solver.factorize()
    solver.solve_local_rhs(tensorProjected)

    return tensorProjected

def calculate_stress_strain(t, dvp_, verbose, visualization_folder, material_parameters, mesh,dx_s, **namespace):

    # Files for storing extra outputs (stresses and strains)
    if not "ep_file" in namespace.keys():
        sig_file = XDMFFile(MPI.comm_world, str(visualization_folder.joinpath("TrueStress.xdmf")))
        pk1_file = XDMFFile(MPI.comm_world, str(visualization_folder.joinpath("PK1Stress.xdmf")))
        ep_file = XDMFFile(MPI.comm_world, str(visualization_folder.joinpath("InfinitesimalStrain.xdmf")))
        for tmp_t in [sig_file,ep_file]:
            tmp_t.parameters["flush_output"] = True
            tmp_t.parameters["rewrite_function_mesh"] = False

        return_dict = dict(ep_file=ep_file,sig_file=sig_file, pk1_file=pk1_file)
    
        namespace.update(return_dict)
    
    else:
        return_dict = {}
    
    # Split function
    d = dvp_["n"].sub(0, deepcopy=True)
  
    # Create tensor function space for stress and strain (this is necessary to evaluate tensor valued functions)
    '''
    Strain/stress are in L2, therefore we use a discontinuous function space with a degree of 1 for P2P1 elements
    Could also use a degree = 0 to get a constant-stress representation in each element
    For more info see the Fenics Book (P62, or P514-515), or
    https://comet-fenics.readthedocs.io/en/latest/demo/viscoelasticity/linear_viscoelasticity.html?highlight=DG#A-mixed-approach
    https://fenicsproject.org/qa/10363/what-is-the-most-accurate-way-to-recover-the-stress-tensor/
    https://fenicsproject.discourse.group/t/why-use-dg-space-to-project-stress-strain/3768
    '''

    Te = TensorElement("DG", mesh.ufl_cell(), 1) 
    Tens = FunctionSpace(mesh, Te)
    
    # Deformation Gradient and first Piola-Kirchoff stress (PK1)
    deformationF = common.F_(d) # calculate deformation gradient from displacement
    
    # Cauchy (True) Stress and Infinitesimal Strain (Only accurate for small strains, ask DB for True strain calculation...)
    epsilon = common.eps(d) # Form for Infinitesimal strain (need polar decomposition if we want to calculate logarithmic/Hencky strain)
    ep = project_solid(epsilon,Tens,dx_s) # Calculate stress tensor
    P_ = common.Piola1(d, material_parameters)  # Form for second PK stress (using St. Venant Kirchoff Model)
    S_ = common.S(d, material_parameters)  # Form for second PK stress (using St. Venant Kirchoff Model)

    sigma = (1/common.J_(d))*deformationF*S_*deformationF.T  # Form for Cauchy (true) stress 

    sig = project_solid(sigma,Tens,dx_s) # Calculate stress tensor
    print("projected True stress tensor")
    PK1 = project_solid(P_,Tens,dx_s) # Calculate stress tensor
    print("projected PK1 stress tensor")
    # Name function
    ep.rename("InfinitesimalStrain", "ep")
    sig.rename("TrueStress", "sig")
    PK1.rename("PK1Stress", "PK1")

    print("Writing Additional Viz Files for Stresses and Strains!")
    # Write results
    namespace["ep_file"].write(ep, t)
    namespace["sig_file"].write(sig, t)
    namespace["pk1_file"].write(PK1, t)

    return return_dict