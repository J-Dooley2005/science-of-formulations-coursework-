
import numpy as np
from scipy.integrate import solve_ivp
from initial_conditions import L, N0, c0, c_sat, rho_c, kv

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


# Calculate initial moments from seed distribution
mu0 = np.sum(N0)
mu1 = np.sum(N0 * L)
mu2 = np.sum(N0 * L**2)
mu3 = np.sum(N0 * L**3)

# Initial conditions
y0 = [mu0, mu1, mu2, mu3, c0]

# Method of moments equations
def moments_model(t, y):

    mu0, mu1, mu2, mu3, c = y

    # Calculate supersaturation
    S = c / c_sat

    # Calculate growth and nucleation rates
    G = growth_rate(S)
    J = nucleation_rate(S) * m_solvent

    # Moment equations
    dmu0dt = J
    dmu1dt = G * mu0
    dmu2dt = 2 * G * mu1
    dmu3dt = 3 * G * mu2

    # Liquid concentration mass balance
    dcdt = -rho_c * kv * dmu3dt / m_solvent

    return [dmu0dt, dmu1dt, dmu2dt, dmu3dt, dcdt]


# Simulation time (24 hours)
t_max = 86400

# Solve method of moments model
solution = solve_ivp(
    moments_model,
    (0, t_max),
    y0,
    method='BDF',
    t_eval=np.linspace(0, t_max, 200),
    rtol=1e-7,
    atol=1e-9
)

# Check solver success
if not solution.success:
    raise RuntimeError(solution.message)

# Extract initial and final moments
mu0_initial = solution.y[0, 0]
mu1_initial = solution.y[1, 0]
mu3_initial = solution.y[3, 0]

mu0_final = solution.y[0, -1]
mu1_final = solution.y[1, -1]
mu3_final = solution.y[3, -1]

# Initial and final liquid concentrations
c_initial = solution.y[4, 0]
c_final = solution.y[4, -1]

# Calculate average crystal lengths
L_initial = mu1_initial / mu0_initial
L_final = mu1_final / mu0_final

# Calculate crystal masses
m_initial = rho_c * kv * mu3_initial
m_final = rho_c * kv * mu3_final

# Check total solute conservation
total_initial = c_initial * m_solvent + m_initial
total_final = c_final * m_solvent + m_final

mass_error = total_final - total_initial

# Print results
print("METHOD OF MOMENTS RESULTS")

print(f"Initial concentration = {c_initial:.4f} g/kg")
print(f"Final concentration = {c_final:.4f} g/kg")

print(f"Initial average length = {L_initial:.2f} um")
print(f"Final average length = {L_final:.2f} um")

print(f"Initial crystal number = {mu0_initial:.2e}")
print(f"Final crystal number = {mu0_final:.2e}")

print(f"Initial crystal mass = {m_initial:.4f} g")
print(f"Final crystal mass = {m_final:.4f} g")

print(f"Mass balance error = {mass_error:.4e} g")
print(f"Final supersaturation = {c_final / c_sat:.4f}")

# Calculate initial and final standard deviations
std_initial = np.sqrt(
    solution.y[2, 0] / solution.y[0, 0]
    - (solution.y[1, 0] / solution.y[0, 0])**2
)

std_final = np.sqrt(
    solution.y[2, -1] / solution.y[0, -1]
    - (solution.y[1, -1] / solution.y[0, -1])**2
)

print(f"Initial standard deviation = {std_initial:.2f} um")
print(f"Final standard deviation = {std_final:.2f} um")