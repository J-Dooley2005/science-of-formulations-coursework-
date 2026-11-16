
import numpy as np

# Initial operating conditions
c0 = 8          # Initial concentration (g/kg)
T = 15          # Temperature (degrees C)

# Solubility parameters
a = 3.37
b = 0.0359

# Calculate solubility
c_sat = a * np.exp(b * T)

# Calculate initial supersaturation
S0 = c0 / c_sat

print(f"Solubility = {c_sat:.4f} g/kg")
print(f"Initial supersaturation = {S0:.4f}")

# Seed crystal parameters
L_mean = 100       # Mean crystal length (um)
L_std = 10         # Standard deviation (um)
m0 = 1             # Initial seed mass (g)

# Crystal properties
rho_c = 1.11e-12   # Crystal density (g/um^3)
kv = np.pi / 4     # Shape factor

# Size discretisation
L_max = 500        # Maximum crystal length (um)
dL = 1             # Bin width (um)

# Create crystal size bins (using bin midpoints)
L = np.arange(dL / 2, L_max, dL)

# Normal distribution of seed lengths
seed_dist = np.exp(-0.5 * ((L - L_mean) / L_std)**2)

# Normalise distribution
seed_dist = seed_dist / np.sum(seed_dist)

# Calculate number of crystals required for 1 g seed mass
N_total = m0 / (rho_c * kv * np.sum(seed_dist * L**3))

# Number of crystals in each size bin
N0 = N_total * seed_dist

# Verify total seed mass
mass_check = rho_c * kv * np.sum(N0 * L**3)

print(f"Total seed crystals = {N_total:.2e}")
print(f"Initial seed mass = {mass_check:.4f} g")
