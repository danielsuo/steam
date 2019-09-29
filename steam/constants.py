import os
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
