import os
import csv
import numpy as np
import pandas as pd

path = os.path.dirname(os.path.abspath(__file__))

# Source: http://www.iapws.org/relguide/IAPWS95-2018.pdf
table1 = pd.read_csv(os.path.join(path, "table1.csv"), delimiter=" ")
table2 = pd.read_csv(os.path.join(path, "table2.csv"), delimiter=" ")

# Critical temperature (K)
T_c = 647.096

# Critical density (kg m^-3)
rho_c = 322

# Constant (kJ kg^-1 K^-1)
R = 0.46151805


# Ideal gas
def phi_0(delta, tau):
    result = np.log(delta) + table1["n^o"][
        0] + table1["n^o"][1] * tau + table1["n^o"][2] * np.log(tau)

    for i in range(3, 8):
        result += table1["n^o"][i] * np.log(
            1 - np.exp(-table1["gamma^o"][i] * tau))

    return result


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
        psi = np.exp(-table2.C[i] * np.power(delta - 1, 2) -
                       table2.D[i] * np.power(tau - 1, 2))
        theta = (1 - tau) + table2.A[i] * np.power(np.power(delta - 1, 2),
                                                   (1 / (2 * table2.beta[i])))
        Delta = np.power(theta, 2) + table2.B[i] * np.power(
            np.power(delta - 1, 2), table2.a[i])
        result += table2.n[i] * np.power(Delta, table2.b[i]) * delta * psi

    return result


# delta = rho / rho_c
# tau = T_c / T
def phi(delta, tau):
    return phi_0(delta, tau) + phi_r(delta, tau)


# rho: mass density
# T: temperature
def f(rho, T):
    return R * T * phi(rho / rho_c, T_c / T)
