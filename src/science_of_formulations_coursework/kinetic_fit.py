
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

popt, pcov = curve_fit(growth_model, S, G)
k1, k2 = popt

print(f"k1 = {k1:.4f}")
print(f"k2 = {k2:.4f}")

# nucleation rate model
def nucleation_model(S, A, B):
    return A * np.exp(-B/(np.log(S))**2)

popt, pcov = curve_fit(nucleation_model, S, J)
A, B = popt

print(f"A = {A:.4e}")
print(f"B = {B:.4f}")

# Calculate predicted values using fitted parameters
G_pred = growth_model(S, k1, k2)
J_pred = nucleation_model(S, A, B)

# Calculate R-squared for both models
R2_growth = 1 - np.sum((G - G_pred)**2) / np.sum((G - np.mean(G))**2)
R2_nucleation = 1 - np.sum((J - J_pred)**2) / np.sum((J - np.mean(J))**2)

print(f"Growth model R² = {R2_growth:.4f}")
print(f"Nucleation model R² = {R2_nucleation:.4f}")

# create graphs for nucleation rate and growth rate
S_fit = np.linspace(S.min(), S.max(), 200)

# calculate model predictions
G_fit = growth_model(S_fit, k1, k2)
J_fit = nucleation_model(S_fit, A, B)

# Growth rate plot
plt.figure(figsize=(8, 5))

plt.scatter(S, G, color='blue', label='Experimental data')
plt.plot(S_fit, G_fit, color='red',
         label=f'Fitted model (R² = {R2_growth:.4f})')

plt.xlabel('Supersaturation, S')
plt.ylabel('Growth rate (μm s⁻¹)')
plt.title('Growth Rate vs Supersaturation')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Nucleation rate plot
plt.figure(figsize=(8, 5))

plt.scatter(S, J, color='blue', label='Experimental data')
plt.plot(S_fit, J_fit, color='red',
         label=f'Fitted model (R² = {R2_nucleation:.4f})')

plt.yscale('log')

plt.xlabel('Supersaturation, S')
plt.ylabel('Nucleation rate (s⁻¹ kg⁻¹)')
plt.title('Nucleation Rate vs Supersaturation')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
