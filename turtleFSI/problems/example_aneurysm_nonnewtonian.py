# Copyright (c) 2025 Simula Research Laboratory
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Problem file for aneurysm FSI simulation
"""
from pathlib import Path
import numpy as np

from vampy.simulation.Womersley import make_womersley_bcs, compute_boundary_geometry_acrn
from vampy.simulation.simulation_common import print_mesh_information
from turtleFSI.problems import *
_turtle_save_files_visualization = save_files_visualization
from dolfin import HDF5File, Mesh, MeshFunction, assemble, FacetNormal, ds, \
    DirichletBC, Measure, inner, parameters, VectorFunctionSpace, Function, XDMFFile, FunctionSpace, MPI
from turtleFSI.modules.nonnewtonian import update_mu_eff
from vasp.simulations.simulation_common import load_probe_points, calculate_and_print_flow_properties, \
    print_probe_points, InterfacePressure, peval, local_project
from turtleFSI.modules.common import J_    

# set compiler arguments
parameters["form_compiler"]["quadrature_degree"] = 6
parameters["form_compiler"]["optimize"] = True
# The "ghost_mode" has to do with the assembly of form containing the facet
# normals n('+') within interior boundaries (dS). for 3D mesh the value should
# be "shared_vertex", for 2D mesh "shared_facet", the default value is "none"
parameters["ghost_mode"] = "shared_vertex"
_compiler_parameters = dict(parameters["form_compiler"])


def set_problem_parameters(default_variables, **namespace):

    # Compute some solid parameters
    # Need to stay here since mus_s and lambda_s are functions of nu_s and E_s
    E_s_val = 1E6
    nu_s_val = 0.45
    mu_s_val = E_s_val / (2 * (1 + nu_s_val))  # 0.345E6
    lambda_s_val = nu_s_val * 2. * mu_s_val / (1. - 2. * nu_s_val)

    default_variables.update(dict(
        # Temporal parameters
        T=2.0,  # Simulation end time
        dt=0.0002,  # Timne step size
        theta=0.501,  # Theta scheme parameter
        save_step=1,  # Save frequency of files for visualisation
        n_save_second_cycle = 500, # Number of steps saved in the second cycle
        checkpoint_step=50,  # Save frequency of checkpoint files
        # Linear solver parameters
        linear_solver="mumps",
        atol=1e-5,  # Absolute tolerance in the Newton solver
        rtol=1e-5,  # Relative tolerance in the Newton solver
        recompute=20,  # Recompute the Jacobian matix within time steps
        recompute_tstep=20,  # Recompute the Jacobian matix over time steps
        # boundary condition parameters
        inlet_id=3,  # inlet id for the fluid
        inlet_outlet_s_id=11,  # inlet and outlet id for solid
        fsi_id=22,  # id for fsi surface
        outer_id=33,  # id for the outer surface of the solid
        # Fluid parameters
        Q_mean=1.69E-06,
        P_mean=11200,
        T_Cycle=1.0,  # Used to define length of flow waveform
        rho_f=1.000E3,  # Fluid density [kg/m3]
        mu_f=3.45E-3,  # Fluid dynamic viscosity [Pa.s]
        dx_f_id=1,  # ID of marker in the fluid domain
        # mesh lifting parameters (see turtleFSI for options)
        extrapolation="laplace",
        extrapolation_sub_type="constant",
        # Solid parameters
        rho_s=1.0E3,  # Solid density [kg/m3]
        mu_s=mu_s_val,  # Solid shear modulus or 2nd Lame Coef. [Pa]
        nu_s=nu_s_val,  # Solid Poisson ratio [-]
        lambda_s=lambda_s_val,  # Solid Young's modulus [Pa]
        dx_s_id=2,  # ID of marker in the solid domain
        k_s=1E5,  # elastic constant of Robin [N/m^3]
        c_s=10,  # viscous constant of Robin [N.s/m^3]
        ds_s_id=33,  # ID of marker for the Robin boundary condition
        robin_bc=True,  # Use Robin boundary condition for the solid
        # Simulation parameters
        folder="sscp_results/C0004/FSI/NonNewtonian",  # Folder name generated for the simulation
        mesh_path="mesh_C0004/model_clipped_3.h5",
        FC_file="FC_MCA_10",  # File name containing the fourier coefficients for the flow waveform
        P_FC_File="FC_Pressure",  # File name containing the fourier coefficients for the pressure waveform
        compiler_parameters=_compiler_parameters,  # Update the defaul values of the compiler arguments (FEniCS)
        save_deg=2,  # Degree of the functions saved for visualisation
        scale_probe=True,  # Scale the probe points to meters
        nonnewtonian = True,
        cross_params = {"mu_0": 0.056, "mu_inf": 0.00345, "lmbda": 3.736, "n": 2.406, "a": 0.254},
        # mu_0=5.6E-2,      # Cross model zero-shear viscosity [Pa.s]
        # mu_inf=3.45E-3,   # Cross model infinite-shear viscosity [Pa.s]
        # lmbda=3.736,      # Cross model time constant [s]
        # n=1.45,
        # a=0.254,
        solid_properties={"dx_s_id":2,"material_model":"MooneyRivlin","rho_s":1.0E3,"mu_s":mu_s_val,"lambda_s":lambda_s_val,"C01":0.02e6,"C10":0.0,"C11":1.8e6},
    ))
    return default_variables


def get_mesh_domain_and_boundaries(mesh_path, **namespace):

    # Read mesh
    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(), mesh_path, "r")
    hdf.read(mesh, "/mesh", False)
    boundaries = MeshFunction("size_t", mesh, 2)
    hdf.read(boundaries, "/boundaries")
    domains = MeshFunction("size_t", mesh, 3)
    hdf.read(domains, "/domains")

    print_mesh_information(mesh)

    return mesh, domains, boundaries

def _save_second_cycle_only(counter: int, dt: float, T_Cycle: float, n_save_second_cycle: int) -> bool:
    """
    Save only selected timesteps during the second cardiac cycle.
    """
    step_number = counter + 1

    steps_per_cycle = int(round(T_Cycle / dt))
    save_stride = steps_per_cycle // n_save_second_cycle

    return (
        step_number > steps_per_cycle
        and (step_number - steps_per_cycle) % save_stride == 0
    )

def save_files_visualization(visualization_folder, dvp_, t, save_deg, v_deg, p_deg, mesh,
                             counter, dt, T_Cycle, n_save_second_cycle, **namespace):
    """
    Wrapper around the default turtleFSI visualization saving.

    It skips the first cardiac cycle and saves only selected timesteps
    during the second cardiac cycle.
    """
    should_save = _save_second_cycle_only(
        counter=counter,
        dt=dt,
        T_Cycle=T_Cycle,
        n_save_second_cycle=n_save_second_cycle,
    )

    step_number = counter + 1
    t_debug = step_number * dt

    if MPI.rank(MPI.comm_world) == 0:
        if counter < 5:
            print(
                f"[DEBUG save_files_visualization] called: "
                f"counter={counter}, step={step_number}, "
                f"t≈{t_debug:.6f}, solver_t={t:.6f}, "
                f"should_save={should_save}",
                flush=True,
            )

        if should_save:
            print(
                f"[DEBUG save_files_visualization] SAVING: "
                f"counter={counter}, step={step_number}, "
                f"t≈{t_debug:.6f}, solver_t={t:.6f}",
                flush=True,
            )

    if not should_save:
        return {}

    return _turtle_save_files_visualization(
        visualization_folder=visualization_folder,
        dvp_=dvp_,
        t=t,
        save_deg=save_deg,
        v_deg=v_deg,
        p_deg=p_deg,
        mesh=mesh,
        **namespace,
    )

def create_bcs(t, DVP, mesh, boundaries, mu_f,
               fsi_id, inlet_id, inlet_outlet_s_id,
               psi, F_solid_linear, p_deg, FC_file,
               Q_mean, P_FC_File,P_mean, T_Cycle, **namespace):

    # Load fourier coefficients for the velocity and scale by flow rate
    An, Bn = np.loadtxt(Path(__file__).parent / FC_file).T
    # Convert to complex fourier coefficients
    Cn = (An - Bn * 1j) * Q_mean
    _, tmp_center, tmp_radius, tmp_normal = compute_boundary_geometry_acrn(mesh, inlet_id, boundaries)

    # Create Womersley boundary condition at inlet
    tmp_element = DVP.sub(1).sub(0).ufl_element()
    inlet = make_womersley_bcs(T_Cycle, None, mu_f, tmp_center, tmp_radius, tmp_normal, tmp_element, Cn=Cn)
    # Initialize inlet expressions with initial time
    for uc in inlet:
        uc.set_t(t)

    # Create Boundary conditions for the velocity
    u_inlet = [DirichletBC(DVP.sub(1).sub(i), inlet[i], boundaries, inlet_id) for i in range(3)]
    u_inlet_s = DirichletBC(DVP.sub(1), ((0.0, 0.0, 0.0)), boundaries, inlet_outlet_s_id)

    # Solid Displacement BCs
    d_inlet = DirichletBC(DVP.sub(0), (0.0, 0.0, 0.0), boundaries, inlet_id)
    d_inlet_s = DirichletBC(DVP.sub(0), (0.0, 0.0, 0.0), boundaries, inlet_outlet_s_id)

    # Assemble boundary conditions
    bcs = u_inlet + [d_inlet, u_inlet_s, d_inlet_s]

    # Load Fourier coefficients for the pressure
    An_P, Bn_P = np.loadtxt(Path(__file__).parent / P_FC_File).T

    # Apply pulsatile pressure at the fsi interface by modifying the variational form
    n = FacetNormal(mesh)
    dSS = Measure("dS", domain=mesh, subdomain_data=boundaries)
    interface_pressure = InterfacePressure(t=0.0, t_ramp_start=0.0, t_ramp_end=0.2, An=An_P,
                                           Bn=Bn_P, period=T_Cycle, P_mean=P_mean, degree=p_deg)
    # NOTE: ('+') implicitly assumes that the solid domain has a higher domain ID than the fluid domain
    F_solid_linear += interface_pressure * inner(n('+'), psi('+')) * dSS(fsi_id)

    # Create inlet subdomain for computing the flow rate inside post_solve
    dsi = ds(inlet_id, domain=mesh, subdomain_data=boundaries)
    inlet_area = assemble(1.0 * dsi)
    return dict(bcs=bcs, inlet=inlet, interface_pressure=interface_pressure, F_solid_linear=F_solid_linear, n=n,
                dsi=dsi, inlet_area=inlet_area)


def initiate(mesh_path, scale_probe, mesh, v_deg, p_deg, **namespace):

    probe_points = load_probe_points(mesh_path)
    probe_points = probe_points[0]
    # In case the probe points are in mm, scale them to meters
    if scale_probe:
        probe_points = probe_points * 0.001

    Vv = VectorFunctionSpace(mesh, "CG", v_deg)

    return dict(probe_points=probe_points)


def pre_solve(t, inlet,interface_pressure, mu_eff, cross_params, v_, d_, dvp_, nonnewtonian, **namespace):
    for uc in inlet:
        # Update the time variable used for the inlet boundary condition
        uc.set_t(t)

        # Multiply by cosine function to ramp up smoothly over time interval 0-250 ms
        if t < 0.25:
            uc.scale_value = -0.5 * np.cos(np.pi * t / 0.25) + 0.5
        else:
            uc.scale_value = 1.0

    # Update pressure condition
    interface_pressure.update(t)

    # update nonewtonian viscosity
    if nonnewtonian:
        has_n_minus_2 = (
            "n-2" in v_
            and "n-2" in d_
            and "n-2" in dvp_
            and dvp_["n-2"].vector().norm("l2") > 0.0
        )

        v2 = v_["n-2"] if has_n_minus_2 else v_["n-1"]
        d2 = d_["n-2"] if has_n_minus_2 else d_["n-1"]

        update_mu_eff(
            mu_eff,
            v_["n-1"],
            d_["n-1"],
            v2,
            d2,
            **cross_params,
        )

    return dict(inlet=inlet, interface_pressure=interface_pressure)

def compute_minimum_jacobian(mesh: Mesh, d: Function, local_rhs: bool = False) -> float:

    # Create a function space for the Jacobian
    DG = FunctionSpace(mesh, "DG", 0)
    # compute the Jacobian of the displacement
    jacobian = J_(d)
    # project the Jacobian to the DG function space
    projected_jacobian = local_project(jacobian, DG, local_rhs)
    min_jacobian = np.min(projected_jacobian.vector().get_local())
    global_min_jacobian = MPI.min(MPI.comm_world, min_jacobian)

    if MPI.rank(MPI.comm_world) == 0:
        print(f"Minimum Jacobian: {global_min_jacobian}")

        if global_min_jacobian <= 0:
            print("Warning: Negative Jacobian detected.")

    return min_jacobian

def post_solve(dvp_, n, dsi, dt, mesh, inlet_area, mu_f, rho_f, probe_points, t,
               mu_eff, visualization_folder,
               counter, save_step, nonnewtonian, MPI,
               T_Cycle, n_save_second_cycle, **namespace):

    d = dvp_["n"].sub(0, deepcopy=True)
    v = dvp_["n"].sub(1, deepcopy=True)
    p = dvp_["n"].sub(2, deepcopy=True)

    # Store viscosity only during the second cardiac cycle
    if nonnewtonian and _save_second_cycle_only(
        counter=counter,
        dt=dt,
        T_Cycle=T_Cycle,
        n_save_second_cycle=n_save_second_cycle,
    ):

        if "mu_file" not in namespace:
            mu_file = XDMFFile(
                MPI.comm_world,
                str(Path(visualization_folder) / "viscosity.xdmf")
            )
            mu_file.parameters["flush_output"] = True
            mu_file.parameters["rewrite_function_mesh"] = False
            namespace["mu_file"] = mu_file

        mu_eff.rename("Viscosity", "mu")
        namespace["mu_file"].write(mu_eff, t)

    print_probe_points(v, p, probe_points)
    calculate_and_print_flow_properties(dt, mesh, v, inlet_area, mu_f, rho_f, n, dsi)
    compute_minimum_jacobian(mesh, d)

    if nonnewtonian and "mu_file" in namespace:
        return dict(mu_file=namespace["mu_file"])

    return dict()
