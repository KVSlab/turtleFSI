# File under GNU GPL (v3) licence, see LICENSE file for details.
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

from turtleFSI.modules import *
from dolfin import ds, MPI

"""
Date: 2022-11-07
Comment from Kei: Naming here is a bit confusing. For example, dx_f, ds_s are hard to understand. 
dx refers to the area of the domain while ds refers to the area of the boundary.
f refere to the fluid while s refers to the solid.
"""

def assign_domain_properties(dx, dx_f_id, rho_f, mu_f, fluid_properties, dx_s_id, material_model, rho_s, mu_s, lambda_s, solid_properties, domains, ds_s_id, boundaries, robin_bc, **namespace):
    """
    Assigns solid and fluid properties to each region.   
    """

    # DB, May 2nd, 2022: All these conversions to lists seem a bit cumbersome, but this allows the solver to be backwards compatible.

    # 1. Create differential for each fluid region, and organize into fluid_properties list of dicts
    dx_f = {}
    if len(fluid_properties) == 0:
        if isinstance(dx_f_id, list): # If dx_f_id is a list (i.e, if there are multiple fluid regions):
            for fluid_region in range(len(dx_f_id)):
                dx_f[fluid_region] = dx(dx_f_id[fluid_region], subdomain_data=domains) # Create dx_f for each fluid region
                fluid_properties.append({"dx_f_id":dx_f_id[fluid_region],"rho_f":rho_f[fluid_region],"mu_f":mu_f[fluid_region]})
            dx_f_id_list=dx_f_id
        else:
            dx_f[0] = dx(dx_f_id, subdomain_data=domains)
            dx_f_id_list=[dx_f_id]
            fluid_properties.append({"dx_f_id":dx_f_id,"rho_f":rho_f,"mu_f":mu_f})
    elif isinstance(fluid_properties, dict):
        fluid_properties = [fluid_properties]

    # Create solid region differentials
    dx_s = {}
    if isinstance(dx_s_id, list): # If dx_s_id is a list (i.e, if there are multiple solid regions):
        for solid_region in range(len(dx_s_id)):
            dx_s[solid_region] = dx(dx_s_id[solid_region], subdomain_data=domains) # Create dx_s for each solid region
        dx_s_id_list=dx_s_id
    else:
        dx_s[0] = dx(dx_s_id, subdomain_data=domains)
        dx_s_id_list=[dx_s_id]

    # Assign material properties to each region
    if len(solid_properties) == 0:
        if isinstance(dx_s_id, list): # If dx_s_id is a list (i.e, if there are multiple solid regions):
            for solid_region in range(len(dx_s_id)):
                if isinstance(material_model, list): 
                    solid_properties.append({"dx_s_id":dx_s_id[solid_region],"material_model":material_model[solid_region],"rho_s":rho_s[solid_region],"mu_s":mu_s[solid_region],"lambda_s":lambda_s[solid_region]})
                else: 
                    solid_properties.append({"dx_s_id":dx_s_id[solid_region],"material_model":material_model,"rho_s":rho_s[solid_region],"mu_s":mu_s[solid_region],"lambda_s":lambda_s[solid_region]})
        else:
            solid_properties.append({"dx_s_id":dx_s_id,"material_model":material_model,"rho_s":rho_s,"mu_s":mu_s,"lambda_s":lambda_s})
    elif isinstance(solid_properties, dict): 
        solid_properties = [solid_properties]
    
    # RobinBC
    if robin_bc:
        ds_s = {}
        if isinstance(ds_s_id, list): # If ds_s_id is a list (i.e, if there are multiple boundary regions):
            for i, solid_boundaries in enumerate(ds_s_id):
                ds_s[i] = ds(solid_boundaries, subdomain_data=boundaries) # Create ds_s for each boundary
            ds_s_ext_id_list=ds_s_id
        else:
            ds_s[0] = ds(ds_s_id, subdomain_data=boundaries)
            ds_s_ext_id_list=[ds_s_id] # If there aren't multpile boundary regions, and the boundary parameters are given as floats, convert solid parameters to lists.
    else:
        ds_s = None
        ds_s_ext_id_list = None

    return dict(dx_f=dx_f, dx_f_id_list=dx_f_id_list, ds_s_ext_id_list=ds_s_ext_id_list, ds_s=ds_s, fluid_properties=fluid_properties, dx_s=dx_s, dx_s_id_list=dx_s_id_list, solid_properties=solid_properties)
