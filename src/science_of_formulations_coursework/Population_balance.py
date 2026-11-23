
import numpy as np
from scipy.integrate import solve_ivp
from initial_conditions import L, dL, N0, S0, c0, c_sat, rho_c, kv
import matplotlib.pyplot as plt

# Number of crystal size bins
n_bins = len(L)

# Fitted growth parameters
k1 = 25.4139
k2 = 1.3195

# Fitted nucleation parameters
A = 1.4e8
B = 10.0

# Assumed solvent mass (kg)
m_solvent = 1.0

# Growth rate as a function of supersaturation
def growth_rate(S):
    if S <= 1:
        return 0.0
    return k1 * (S - 1)**k2

# Nucleation rate as a function of supersaturation
def nucleation_rate(S):
    if S <= 1:
        return 0.0
    return A * np.exp(-B / (np.log(S))**2)


# Initial growth and nucleation rates
G_initial = growth_rate(S0)
J_initial = nucleation_rate(S0)

print(f"Initial growth rate = {G_initial:.4f} um/s")
print(f"Initial nucleation rate = {J_initial:.4e} crystals/(s kg)")


# Coupled population balance and liquid mass balance
def pbe(t, y):

    # Separate crystal populations and liquid concentration
    N = y[:n_bins]
    c = y[-1]

    # Calculate current supersaturation
    S = c / c_sat

    # Calculate growth and nucleation rates
    G = growth_rate(S)
    J = nucleation_rate(S)

    # Population balance
    dNdt = np.zeros(n_bins)

    # Nucleation adds new crystals to the first bin
    dNdt[0] = J * m_solvent - G * N[0] / dL

    # Growth transfers crystals between bins
    dNdt[1:] = G * (N[:-1] - N[1:]) / dL

    # Liquid concentration mass balance
    dcdt = -rho_c * kv * np.sum(L**3 * dNdt) / m_solvent

    # Return derivatives for solver
    return np.concatenate((dNdt, [dcdt]))


# Initial conditions for coupled model
y0 = np.concatenate((N0, [c0]))

# Full reference simulation time (24 hours)
t_max = 86400

# Solve coupled PBE and mass balance
solution = solve_ivp(
    pbe,
    (0, t_max),
    y0,
    method='BDF',
    t_eval=np.linspace(0, t_max, 100),
    rtol=1e-7,
    atol=1e-9
)

# Check solver success
if not solution.success:
    raise RuntimeError(solution.message)


# Extract results
N_initial = solution.y[:n_bins, 0]
N_final = solution.y[:n_bins, -1]

c_initial = solution.y[-1, 0]
c_final = solution.y[-1, -1]

# Calculate average crystal lengths
L_initial = np.sum(N_initial * L) / np.sum(N_initial)
L_final = np.sum(N_final * L) / np.sum(N_final)

# Calculate crystal masses
m_initial = rho_c * kv * np.sum(N_initial * L**3)
m_final = rho_c * kv * np.sum(N_final * L**3)

# Calculate total number of crystals
N_total_initial = np.sum(N_initial)
N_total_final = np.sum(N_final)

# Check overall solute conservation
total_initial = c_initial * m_solvent + m_initial
total_final = c_final * m_solvent + m_final

mass_error = total_final - total_initial

# Print simulation results
print(f"Number of bins = {n_bins}")

print(f"Initial concentration = {c_initial:.4f} g/kg")
print(f"Final concentration = {c_final:.4f} g/kg")

print(f"Initial average length = {L_initial:.2f} um")
print(f"Final average length = {L_final:.2f} um")

print(f"Initial crystal number = {N_total_initial:.2e}")
print(f"Final crystal number = {N_total_final:.2e}")

print(f"Initial crystal mass = {m_initial:.4f} g")
print(f"Final crystal mass = {m_final:.4f} g")

print(f"Initial total solute = {total_initial:.4f} g")
print(f"Final total solute = {total_final:.4f} g")
print(f"Mass balance error = {mass_error:.4e} g")

print(f"Final supersaturation = {c_final / c_sat:.4f}")


# Validate crystal size distribution

# Calculate standard deviation of crystal lengths
std_initial = np.sqrt(
    np.sum(N_initial * (L - L_initial)**2)
    / np.sum(N_initial)
)

std_final = np.sqrt(
    np.sum(N_final * (L - L_final)**2)
    / np.sum(N_final)
)

print(f"Initial size standard deviation = {std_initial:.2f} um")
print(f"Final size standard deviation = {std_final:.2f} um")

# Check crystal number conservation
number_error = N_total_final - N_total_initial
print(f"Crystal number change = {number_error:.4e}")

# Check whether crystals reach maximum size
boundary_fraction = N_final[-1] / np.sum(N_final)
print(f"Fraction in final size bin = {boundary_fraction:.4e}")

# Check mass conservation throughout simulation
crystal_mass = rho_c * kv * np.sum(
    solution.y[:n_bins, :] * L[:, None]**3, axis=0
)

total_solute = solution.y[-1, :] * m_solvent + crystal_mass
max_mass_error = np.max(np.abs(total_solute - total_initial))

print(f"Maximum mass balance error = {max_mass_error:.4e} g")

# Plot initial and final crystal size distributions
plt.figure(figsize=(8, 5))

plt.plot(
    L, N_initial / np.sum(N_initial),
    label='Initial distribution',
    color='blue'
)

plt.plot(
    L, N_final / np.sum(N_final),
    label='Final distribution',
    color='red'
)

plt.xlabel('Crystal length (um)')
plt.ylabel('Fraction of crystals per size bin')
plt.title('Crystal Size Distribution')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Time array
time = solution.t

# Liquid concentration over time
c_time = solution.y[-1, :]

# Solid crystal mass over time
solid_mass_time = rho_c * kv * np.sum(
    solution.y[:n_bins, :] * L[:, None]**3,
    axis=0
)

# Plot solid mass vs time
plt.figure(figsize=(8, 5))
plt.plot(time, solid_mass_time)
plt.xlabel('Time (s)')
plt.ylabel('Solid crystal mass (g)')
plt.title('Solid Mass vs Time')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot liquid concentration vs time
plt.figure(figsize=(8, 5))
plt.plot(time, c_time)
plt.xlabel('Time (s)')
plt.ylabel('Liquid concentration (g/kg)')
plt.title('Liquid Concentration vs Time')
plt.grid(True)
plt.tight_layout()
plt.show()