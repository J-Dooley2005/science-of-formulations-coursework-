import pandas as pd
import numpy as np 
from scipy.optimize import curve_fit

data = pd.read_csv("KineticData.csv")

# extract the columns
S = data['Supersaturation [-]']
G = data['Growth rate [μm s^{-1}]']


# Growth model function
def growth_model(S, k1, k2):
    return k1 * (S-1)**k2

potp, pcov = curve_fit(growth_model, S, G)
k1, k2 = potp

print(f"k1 = {k1:.4f}")
print(f"k2 = {k2:.4f}")