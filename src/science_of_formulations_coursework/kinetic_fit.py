import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

data = pd.read_csv("KineticData.csv")

# extract the columns
S = data['Supersaturation [-]']
G = data['Growth rate [μm s^{-1}]']
J = data['Nucleation rate [s^{-1} kg^{-1}]']


# Growth model function
def growth_model(S, k1, k2):
    return k1 * (S-1)**k2

potp, pcov = curve_fit(growth_model, S, G)
k1, k2 = potp

print(f"k1 = {k1:.4f}")
print(f"k2 = {k2:.4f}")

# nucleation rate model
def nucleation_model(S, A, B):
    return A * np.exp(-B/(np.log(S))**2)

potp, pcov = curve_fit(nucleation_model, S, J)
A, B = potp

print(f"A = {A:.4e}")
print(f"B = {B:.4f}")

# create graphs for nucleation rate and growth rate

S_fit = np.linspace(S.min(), S.max(), 200)

# calculate model predictions
G_fit = growth_model(S_fit, k1, k2)
J_fit = nucleation_model(S_fit, A, B)

#Growth rate plot
plt.figure(figsize=(8,5))
plt.plot(S_fit, G_fit, color='red', label='Fitted model')

plt.xlabel('Supersaturation, S')
plt.ylabel('Growth rate (μm s^{-1} )')
plt.title('Growth rate vs Supersaturation')
plt.legend()
plt.grid(True)

plt.show()

# nucleation rate plot
plt.figure(figsize=(8,5))
plt.plot(S_fit, J_fit, color='red', label='Fitted model')

plt.yscale('log')  # Important for nucleation data

plt.xlabel('Supersaturation, S')
plt.ylabel('Nucleation Rate (s⁻¹ kg⁻¹)')
plt.title('Nucleation Rate vs Supersaturation')
plt.legend()
plt.grid(True)
plt.show()