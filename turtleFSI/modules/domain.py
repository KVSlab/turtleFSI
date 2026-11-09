# File under GNU GPL (v3) licence, see LICENSE file for details.
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

from turtleFSI.modules import *
from dolfin import ds, MPI
from dolfin.cpp.mesh import MeshFunctionSizet
from typing import Union
import numpy as np

"""
Last update: 2023-06-15
Kei Yamamoto added docstring for assign_domain_properties function and added comments in the function.
"""

def assign_domain_properties(dx: ufl.measure.Measure, dx_f_id: Union[int, list], rho_f: Union[float, list], 
                             mu_f: Union[float, list], fluid_properties: Union[list, dict], dx_s_id: Union[int, list], 
                             material_model: str, rho_s: Union[float, list], mu_s: Union[float, list], lambda_s: Union[float, list],
                             solid_properties: Union[list, dict], domains: MeshFunctionSizet, ds_s_id: Union[int, list],
                             boundaries: MeshFunctionSizet, robin_bc: bool, **namespace):
    """
    Assigns solid and fluid properties to each region.   

    Args:
        dx: Measure of the domain 
        dx_f_id: ID of the fluid region, or list of IDs of multiple fluid regions
        rho_f: Density of the fluid, or list of densities of multiple fluid regions
        mu_f: Viscosity of the fluid, or list of viscosities of multiple fluid regions
        fluid_properties: Dictionary of fluid properties, or list of dictionaries of fluid properties for multiple fluid regions
        dx_s_id: ID of the solid region, or list of IDs of multiple solid regions
        material_model: Material model of the solid, or list of material models of multiple solid regions
        rho_s: Density of the solid, or list of densities of multiple solid regions
        mu_s: Shear modulus or 2nd Lame Coef. of the solid, or list of shear modulus of multiple solid regions
        lambda_s: First Lame parameter of the solid, or list of first Lame parameters of multiple solid regions
        solid_properties: Dictionary of solid properties, or list of dictionaries of solid properties for multiple solid regions
        domains: MeshFunction of the domains
        ds_s_id: ID of the solid boundary, or list of IDs of multiple solid boundaries where Robin boundary conditions are applied
        boundaries: MeshFunction of the boundaries
        robin_bc: True if Robin boundary conditions are used, False otherwise

    Returns:
        dx_f: Measure of the fluid domain for each fluid region
        dx_f_id_list: List of IDs of single/multiple fluid regions
        ds_s_ext_id_list: List of IDs of single/multiple solid boundaries where Robin boundary conditions are applied
        ds_s: Measure of the solid boundary for each solid region
        fluid_properties: List of dictionaries of fluid properties for single/multiple fluid regions
        dx_s: Measure of the solid domain for each solid region
        dx_s_id_list: List of IDs of single/multiple solid regions
        solid_properties: List of dictionaries of solid properties for single/multiple solid regions

    """
    # DB, May 2nd, 2022: All these conversions to lists seem a bit cumbersome, but this allows the solver to be backwards compatible.
    # Work on fluid domain 
    dx_f = {}
    # In case there are multiple fluid regions, we assume that dx_f_id is a list
    if isinstance(dx_f_id, list):
        for fluid_region in range(len(dx_f_id)):
            dx_f[fluid_region] = dx(dx_f_id[fluid_region], subdomain_data=domains) # Create dx_f for each fluid domain
        dx_f_id_list=dx_f_id
    # In case there is only one fluid region, we assume that dx_f_id is an int
    else:
        dx_f[0] = dx(dx_f_id, subdomain_data=domains)
        dx_f_id_list=[dx_f_id]
    # Check if fluid_porperties is empty and if so, create fluid_properties to each region, 
    if len(fluid_properties) == 0:
        if isinstance(dx_f_id, list): 
            for fluid_region in range(len(dx_f_id)):
                fluid_properties.append({"dx_f_id":dx_f_id[fluid_region],"rho_f":rho_f[fluid_region],"mu_f":mu_f[fluid_region]})
        else:    
            fluid_properties.append({"dx_f_id":dx_f_id,"rho_f":rho_f,"mu_f":mu_f})
    # If fluid_properties is not empty, assume that fluid_properties is given and convert it to a list if it is not a list
    elif isinstance(fluid_properties, dict):
        fluid_properties = [fluid_properties]
    else:
        assert isinstance(fluid_properties, list), "fluid_properties must be a list of dictionaries"

    # Work on solid domain and boundary (boundary is only needed if Robin boundary conditions are used)    
    dx_s = {}
    # In case there are multiple solid regions, we assume that dx_s_id is a list
    if isinstance(dx_s_id, list):
        for solid_region in range(len(dx_s_id)):
            dx_s[solid_region] = dx(dx_s_id[solid_region], subdomain_data=domains) # Create dx_s for each solid domain
        dx_s_id_list=dx_s_id
    else:
        dx_s[0] = dx(dx_s_id, subdomain_data=domains)
        dx_s_id_list=[dx_s_id]

    # Assign material properties to each solid region
    # NOTE: len(solid_properties) == 0 only works for St. Venant-Kirchhoff material model. 
    #       For other material models, solid_properties must be given from config file or inside the problem file.
    if len(solid_properties) == 0:
        if isinstance(dx_s_id, list): 
            for solid_region in range(len(dx_s_id)):
                if isinstance(material_model, list): 
                    solid_properties.append({"dx_s_id":dx_s_id[solid_region],"material_model":material_model[solid_region],"rho_s":rho_s[solid_region],"mu_s":mu_s[solid_region],"lambda_s":lambda_s[solid_region]})
                else: 
                    solid_properties.append({"dx_s_id":dx_s_id[solid_region],"material_model":material_model,"rho_s":rho_s[solid_region],"mu_s":mu_s[solid_region],"lambda_s":lambda_s[solid_region]})
        else:
            solid_properties.append({"dx_s_id":dx_s_id,"material_model":material_model,"rho_s":rho_s,"mu_s":mu_s,"lambda_s":lambda_s})
    elif isinstance(solid_properties, dict): 
        solid_properties = [solid_properties]
    else:
        assert isinstance(solid_properties, list), "solid_properties must be a list of dictionaries"

    # Create solid boundary differentials for Robin boundary conditions. 
    if robin_bc:
        ds_s = {}
        # In case there are multiple solid boundaries, we assume that ds_s_id is a list and create ds_s for each solid boundary
        if isinstance(ds_s_id, list):
            for i, solid_boundaries in enumerate(ds_s_id):
                ds_s[i] = ds(solid_boundaries, subdomain_data=boundaries) 
            ds_s_ext_id_list=ds_s_id
        else:
            ds_s[0] = ds(ds_s_id, subdomain_data=boundaries)
            ds_s_ext_id_list=[ds_s_id] 
    # If Robin boundary conditions are not used, set ds_s and ds_s_ext_id_list to None.
    else:
        ds_s = None
        ds_s_ext_id_list = None

    return dict(dx_f=dx_f, dx_f_id_list=dx_f_id_list, ds_s_ext_id_list=ds_s_ext_id_list, ds_s=ds_s, fluid_properties=fluid_properties, dx_s=dx_s, dx_s_id_list=dx_s_id_list, solid_properties=solid_properties)


# Parameters of each solid material model, in the order they are printed (see Piola1 in turtleFSI/modules/common.py)
_SOLID_MODEL_PARAMETERS = {
    "StVenantKirchoff": ("rho_s", "mu_s", "lambda_s"),
    "StVenantKirchoffEnergy": ("rho_s", "mu_s", "lambda_s"),
    "NeoHookean": ("rho_s", "mu_s", "lambda_s"),
    "MooneyRivlin": ("rho_s", "mu_s", "lambda_s", "C01", "C10", "C11"),
    "Gent": ("rho_s", "mu_s", "Jm"),
}


def _as_list(value) -> list:
    """Return value as a list: None -> [], a list or tuple -> list, anything else -> [value]."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _format_parameters(parameters: dict, keys: list) -> str:
    """Format the given keys of a dictionary as 'key = value' pairs."""
    pairs = []
    for key in keys:
        value = parameters[key]
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            pairs.append(f"{key} = {value:.4e}")
        else:
            pairs.append(f"{key} = {value}")
    return ", ".join(pairs)


def print_material_models(solid_properties: list, fluid_properties: list, solid: str = "solid", fluid: str = "fluid",
                          robin_bc: bool = False, k_s=None, c_s=None, ds_s_ext_id_list=None,
                          nonnewtonian: bool = False, cross_params=None, **namespace) -> None:
    """
    Print the material model and the parameters of each solid and fluid region, and the Robin boundary condition,
    as they are used in the simulation (i.e. after assign_domain_properties).

    Args:
        solid_properties: List of dictionaries of solid properties for each solid region
        fluid_properties: List of dictionaries of fluid properties for each fluid region
        solid: Solid module ("solid" or "no_solid")
        fluid: Fluid module ("fluid" or "no_fluid")
        robin_bc: True if the Robin boundary condition is used for the solid
        k_s, c_s: Elastic and viscous coefficients of the Robin boundary condition, one per boundary
        ds_s_ext_id_list: IDs of the boundaries where the Robin boundary condition is applied
        nonnewtonian: Default non-Newtonian flag for the fluid regions that do not set it
        cross_params: Default modified Cross parameters for the fluid regions that do not set them
    """
    if MPI.rank(MPI.comm_world) != 0:
        return

    width = 80
    lines = ["", " Material models ".center(width, "=")]

    if solid == "no_solid":
        lines.append("Solid: not solved (solid = no_solid)")
    else:
        for region_properties in _as_list(solid_properties):
            model = region_properties.get("material_model")
            lines.append(f"Solid region dx_s_id={region_properties.get('dx_s_id', '?')}: {model}")
            keys = list(_SOLID_MODEL_PARAMETERS.get(model, ()))
            keys += [key for key in region_properties if key not in keys and key not in ("dx_s_id", "material_model")]
            keys = [key for key in keys if key in region_properties]
            lines.append("    " + _format_parameters(region_properties, keys))

    if fluid == "no_fluid":
        lines.append("Fluid: not solved (fluid = no_fluid)")
    else:
        for region_properties in _as_list(fluid_properties):
            region = region_properties.get("dx_f_id", "?")
            if region_properties.get("nonnewtonian", nonnewtonian):
                params = region_properties.get("cross_params", cross_params) or {}
                lines.append(f"Fluid region dx_f_id={region}: non-Newtonian (modified Cross)")
                text = _format_parameters(region_properties, [key for key in ("rho_f",) if key in region_properties])
                lines.append("    " + ", ".join(part for part in (text, _format_parameters(params, list(params))) if part))
            else:
                lines.append(f"Fluid region dx_f_id={region}: Newtonian")
                lines.append("    " + _format_parameters(region_properties,
                                                         [key for key in ("rho_f", "mu_f") if key in region_properties]))

    if solid != "no_solid":
        if robin_bc:
            lines.append("Robin boundary condition: on")
            for ds_id, k, c in zip(_as_list(ds_s_ext_id_list), _as_list(k_s), _as_list(c_s)):
                lines.append(f"    ds_s_id = {ds_id}: " + _format_parameters({"k_s": k, "c_s": c}, ["k_s", "c_s"]))
        else:
            lines.append("Robin boundary condition: off")

    lines.append("=" * width)
    print("\n".join(lines), flush=True)
