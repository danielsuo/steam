import os
import csv
import jax.numpy as np
import pandas as pd

from .constants import table1, table2, T_c, rho_c, R

# Ideal gas
def phi_o(delta, tau):
    result = np.log(delta) + table1["n^o"][
        0] + table1["n^o"][1] * tau + table1["n^o"][2] * np.log(tau)

    for i in range(3, 8):
        result += table1["n^o"][i] * np.log(
            1 - np.exp(-table1["gamma^o"][i] * tau))

    return result

def psi(delta, tau, i):
    return np.exp(-table2.C[i] * np.power(delta - 1, 2) - table2.D[i] * np.power(tau - 1, 2))

def theta(delta, tau, i):
    return (1 - tau) + table2.A[i] * np.power(np.power(delta - 1, 2), (1 / (2 * table2.beta[i])))

def Delta(delta, tau, i):
    return np.power(theta(delta, tau, i), 2) + table2.B[i] * np.power(np.power(delta - 1, 2), table2.a[i])

# Residual
def phi_r(delta, tau):
    result = 0

    for i in range(0, 7):
        result += table2.n[i] * np.power(delta, table2.d[i]) * np.power(
            tau, table2.t[i])

    for i in range(7, 51):
        # TODO: gamma?
        result += table2.n[i] * np.power(delta, table2.d[i]) * np.power(
            tau, table2.t[i]) * np.exp(-np.power(delta, table2.c[i]))

    for i in range(51, 54):
        result += table2.n[i] * np.power(delta, table2.d[i]) * np.power(
            tau, table2.t[i]) * np.exp(
                -table2.alpha[i] * np.power(delta - table2.epsilon[i], 2) -
                table2.beta[i] * np.power(tau - table2.gamma[i], 2))

    for i in range(54, 56):
        result += table2.n[i] * np.power(Delta(delta, tau, i), table2.b[i]) * delta * psi(delta, tau, i)

    return result


# delta = rho / rho_c
# tau = T_c / T
def phi(delta, tau):
    return phi_o(delta, tau) + phi_r(delta, tau)


# rho: mass density
# T: temperature
def f(rho, T):
    return R * T * phi(rho / rho_c, T_c / T)
