import dolfin as dl
from turtleFSI.modules.common import F_

def compute_shear_rate(v, d):
    l = dl.grad(v) * dl.inv(F_(d))
    D = 0.5 * (l + l.T)
    return dl.sqrt(2 * dl.inner(D, D))

def modified_cross_model(shear_rate,
                        mu_0: float,
                        mu_inf: float,
                        lmbda: float,
                        n: float,
                        a: float):
    
    return mu_inf + (mu_0 - mu_inf) / (1 + (lmbda * shear_rate)**n)**a


def update_mu_eff(old_mu_eff: dl.Function, v_1: dl.Function, d_1: dl.Function, v_2: dl.Function, d_2: dl.Function,
                  mu_0: float, mu_inf: float, lmbda: float,
                  n: float, a: float) -> None:
    """
    Updates old_mu_eff in place with the new effective viscosity. Second order explicit.
    """
    shear_rate_1 = compute_shear_rate(v_1, d_1)
    shear_rate_2 = compute_shear_rate(v_2, d_2)

    up_mu_eff_1 = modified_cross_model(shear_rate_1, mu_0, mu_inf, lmbda, n, a)
    up_mu_eff_2 = modified_cross_model(shear_rate_2, mu_0, mu_inf, lmbda, n, a)

    up_mu_eff = (3/2)*up_mu_eff_1 - (1/2)*up_mu_eff_2
    old_mu_eff.assign(dl.project(up_mu_eff, old_mu_eff.function_space()))

    # clipping part
    vals = old_mu_eff.vector()[:]
    vals[:] = vals.clip(mu_inf, mu_0)
    old_mu_eff.vector()[:] = vals
