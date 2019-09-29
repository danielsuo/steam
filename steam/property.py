from jax import grad

from .constants import table1, table2, T_c, rho_c, R
from .helmholtz import phi_o, phi_r

# Pressure
def p(delta, tau):
    return delta * rho_c * R * T_c / tau * (1 + delta * grad(phi_r, 0)(delta, tau))

# Internal energy
def u(delta, tau):
    return tau * R * T_c / tau * (grad(phi_o, 1)(delta, tau) + grad(phi_r, 1)(delta, tau))

# Entropy
def s(delta, tau):
    return R * (tau * (grad(phi_o, 1)(delta, tau) + grad(phi_r, 1)(delta, tau)) - phi_o(delta, tau) - phi_r(delta, tau))

# Enthalpy
def h(delta, tau):
    return R * T_c / tau * (1 + tau * (grad(phi_o, 1)(delta, tau) + grad(phi_r, 1)(delta, tau)) + delta * grad(phi_r, 0)(delta, tau))
