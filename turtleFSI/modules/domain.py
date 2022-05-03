# File under GNU GPL (v3) licence, see LICENSE file for details.
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

from turtleFSI.modules import *
from dolfin import Constant, inner, inv, grad, div


def assign_domain_properties(dx, dx_f_id, rho_f, mu_f, dx_s_id, rho_s, mu_s, lambda_s, domains, **namespace):
    """
    Assigns solid and fluid properties to each region.   
    """

    # DB, May 2nd, 2022: All these conversions to lists seem a bit cumbersome, but this allows the solver to be backwards compatible.
    dx_f = {}
    if isinstance(dx_f_id, list): # If dx_f_id is a list (i.e, if there are multiple fluid regions):
        for fluid_region in range(len(dx_f_id)):
            dx_f[fluid_region] = dx(dx_f_id[fluid_region], subdomain_data=domains) # Create dx_f for each fluid region
        mu_f_list=mu_f # 
        dx_f_id_list=dx_f_id
    else:
        dx_f[0] = dx(dx_f_id, subdomain_data=domains)
        mu_f_list=[mu_f] # If there aren't multpile fluid regions, and the fluid viscosity is given as a float,convert to list.
        dx_f_id_list=[dx_f_id]
    
    dx_s = {}

    if isinstance(dx_s_id, list): # If dx_s_id is a list (i.e, if there are multiple solid regions):
        for solid_region in range(len(dx_s_id)):
            dx_s[solid_region] = dx(dx_s_id[solid_region], subdomain_data=domains) # Create dx_s for each solid region
        rho_s_list=rho_s
        mu_s_list=mu_s
        lambda_s_list=lambda_s
        dx_s_id_list=dx_s_id
    else:
        dx_s[0] = dx(dx_s_id, subdomain_data=domains)
        rho_s_list=[rho_s] # If there aren't multpile solid regions, and the solid parameters are given as floats, convert solid parameters to lists.
        mu_s_list=[mu_s]
        lambda_s_list=[lambda_s]
        dx_s_id_list=[dx_s_id]


    return dict(dx_f=dx_f, dx_f_id_list=dx_f_id_list, mu_f_list=mu_f_list, dx_s=dx_s, dx_s_id_list=dx_s_id_list, rho_s_list=rho_s_list, mu_s_list=mu_s_list, lambda_s_list=lambda_s_list)
