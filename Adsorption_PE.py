""" PARAMETER ESTIMATION MODEL - ADSORPTION STAGE"""

"""

Article: Kinetic modelling of the CO2 capture and utilisation on NiRu-Ca/Al dual function material via parameter estimation
Authors: Meshkat Dolat, Andrew Wright, Soudabeh Bahrami Gharamaleki, Loukia-Pantzechroula Merkouri, Melis S. Duyar, Michael Short
m.short@surrey.ac.uk

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import perf_counter
from scipy.optimize import fsolve
import optuna
from optuna.samplers import TPESampler


# Parameters
Di = 2.216                      # Reactor (Tube) Inside Diameter (cm)                 
Ai = 3.14 * (Di**2) / 4         # Reactor (Tube) Internal Surface Area (cm2)
L = 1                           # Length of the Reactor (tube) (cm)
u = 1.95                        # Linear velocity (cm/s) 
W_DFM = 2                       # Weight of DFM in the reactor (gr) 
V_react = Ai * L                # Reactor (Tube) Volue (mL)
rho = W_DFM/V_react             # Adsorption bed Density (g/cm^3)                 

Omega = 0.448               # Maximum adsorption capacity (mmol/g)
Alfa = 0.8                  # Adsorption strength correction factor
D = 0.16                    # Diffusion coefficient (cm^2/s)
epsilon = 0.35              # Porosity 
R = 8.314e-3                # Ideal gas constant (J/K.mmol)
R_ = 0.08206                # (L.atm / mol.K)
T = 653                     # Temperature (K) (Assuming isothermal operation)
P_ads = 1                   # Total pressure during adsorption (atm)
P_prg = 1                   # Total pressure during purge (atm)
P_hyd = 1                   # Total pressure during hydrogenation (atm)  

Time_ads = 1200             # Total simulation time (s) for adsorption 
Time_prg = 900              # Total simulation time (s) for purge           
Time_hyd = 300              # Total simulation time (s) for hydrogenation 
Time_prg2 = 120             # Total simulation time (s) for the 2nd purge 
Nx = 5                      # Number of spatial grid points
Nt_ads = 3000               # Number of time steps
Nt_prg = 2000               # Number of time steps
Nt_hyd = 5000               # Number of time steps
Nt_prg2 = 2000              # Number of time steps

C_feed_ads_CO2 = 12.2       # CO2 %vol in feed gas - Adsorption stage (mmol/cm^3)  
C_feed_ads_H2O = 0          # H2O %vol in feed gas - Adsorption stage (mmol/cm^3)
C_feed_purge_CO2 = 0        # CO2 %vol in feed gas - Purge stage (mmol/cm^3)
C_feed_purge_H2O = 0        # H2O %vol in feed gas - Purge stage (mmol/cm^3)
C_feed_hyd_H2 = 10         # H2  %vol in feed gas - Hydrogenation stage (mmol/cm^3) 
C_feed_hyd_N2 = 100 - C_feed_hyd_H2        # N2  %vol in feed gas - Hydrogenation stage (mmol/cm^3) 

# Analyzer delay response parameters Optimized:
tau_rise = 182.9
zeta_rise = 0.76
tau_fall = 222.4
zeta_fall = 0.78

# Load experimental data
experimental_data_CO2_ads = pd.read_excel('data/Adsorption_Data_delagged.xlsx')

time_ads_exp_CO2 = experimental_data_CO2_ads.iloc[:, 0].values
concentration_ads_exp_CO2 = experimental_data_CO2_ads.iloc[:, 1].values

# Function to simulate the model with adsorption of CO2 and H2O
def simulate_adsorption_model(k1, k2, k3, T, P_ads, D, L, Time_ads, Nx, Nt, u, epsilon, C_feed_ads_CO2, C_feed_ads_H2O, rho, Omega):
    dx = L / (Nx - 1)
    dt = Time_ads / Nt

    C_CO2 = np.zeros((Nt + 1, Nx))
    C_H2O = np.zeros((Nt + 1, Nx))
    theta_CO2 = np.zeros((Nt + 1, Nx))
    theta_H2O = np.zeros((Nt + 1, Nx))

    R_ = 0.08206
    C_CO2_init = (C_feed_ads_CO2 / 100) * P_ads / (R_ * T)
    C_H2O_init = (C_feed_ads_H2O / 100) * P_ads / (R_ * T)

    C_CO2[0, 0] = C_CO2_init
    C_H2O[0, 0] = C_H2O_init
    theta_H2O[0, :] = 0.5           # Initial coverage factor of H2O
    theta_CO2[0, :] = 0          # Initial coverage factor of CO2

    def backward_euler_equations(y, i):
        C_CO2_next, C_H2O_next, theta_CO2_next, theta_H2O_next = y
        
        # Diffusion terms
        if i == Nx - 1:  # Outlet boundary condition
            diffusion_CO2 = (D * dt / epsilon) * (-C_CO2_next + C_CO2[t, i - 1]) / dx**2
            diffusion_H2O = (D * dt / epsilon) * (-C_H2O_next + C_H2O[t, i - 1]) / dx**2
        elif i == 1:  # Inlet boundary condition
            diffusion_CO2 = (D * dt / epsilon) * (C_CO2[t, i + 1] - C_CO2_next) / dx**2
            diffusion_H2O = (D * dt / epsilon) * (C_H2O[t, i + 1] - C_H2O_next) / dx**2
        else:  # Internal nodes
            diffusion_CO2 = (D * dt / epsilon) * (C_CO2[t, i + 1] - 2 * C_CO2_next + C_CO2[t, i - 1]) / dx**2
            diffusion_H2O = (D * dt / epsilon) * (C_H2O[t, i + 1] - 2 * C_H2O_next + C_H2O[t, i - 1]) / dx**2

        # Convection terms
        convection_CO2 = - (u * dt / epsilon) * (C_CO2_next - C_CO2[t, i - 1]) / dx
        convection_H2O = - (u * dt / epsilon) * (C_H2O_next - C_H2O[t, i - 1]) / dx

        # Reaction terms
        CO2_formation_rate = - k1 * C_CO2_next * (1 - theta_CO2_next - theta_H2O_next) - k2 * C_CO2_next * theta_H2O_next
        H2O_formation_rate = (
            k2 * C_CO2_next * theta_H2O_next - k3 * C_H2O_next * (1 - theta_CO2_next - theta_H2O_next)
        )

        # Implicit equations
        eq1 = C_CO2_next - C_CO2[t, i] - diffusion_CO2 - convection_CO2 - (rho * CO2_formation_rate * dt / epsilon)
        eq2 = C_H2O_next - C_H2O[t, i] - diffusion_H2O - convection_H2O - (rho * H2O_formation_rate * dt / epsilon)
        eq3 = theta_CO2_next - theta_CO2[t, i] + (CO2_formation_rate * dt / Omega)
        eq4 = theta_H2O_next - theta_H2O[t, i] + (H2O_formation_rate * dt / Omega)

        return [eq1, eq2, eq3, eq4]


    for t in range(0,Nt):
        # Apply boundary conditions for inlet
        C_CO2[t + 1, 0] = C_CO2_init
        C_H2O[t + 1, 0] = C_H2O_init
        # Update coverage factors

        for i in range(1, Nx):
            initial_guess = [C_CO2[t, i], C_H2O[t, i], theta_CO2[t, i], theta_H2O[t, i]]
            solution = fsolve(backward_euler_equations, initial_guess, args=(i,))
            C_CO2[t + 1, i], C_H2O[t + 1, i], theta_CO2[t + 1, i], theta_H2O[t + 1, i] = solution
        
        theta_CO2[t + 1, i] = np.clip(theta_CO2[t + 1, i], 0, 1)
        theta_H2O[t + 1, i] = np.clip(theta_H2O[t + 1, i], 0, 1)

    return C_CO2, C_H2O, theta_CO2, theta_H2O


def convert_mmol_per_mL_to_percent(C_mmol_per_mL, T, P):
    """
    Convert concentration from mmol/mL to volume %.

    Parameters:
    - C_mmol_per_mL: Array or scalar concentration in mmol/mL
    - T: Temperature (K)
    - P: Pressure (atm)

    Returns:
    - C_percent: Converted concentration in %
    """
    R = 0.08206  # Gas constant in L·atm / mol·K
    C_percent = (C_mmol_per_mL * R * T / P) * 100
    return C_percent


def apply_analyzer_delay_2nd_order(C_model, dt, tau_rise, zeta_rise, tau_fall, zeta_fall):
    """
    Applies a 2nd-order response model to smooth the CH4 concentration profile, 
    dynamically adjusting tau and zeta based on the analyzer output trend.
    
    Parameters:
    - C_model: Array of model-predicted CH4 concentration over time
    - dt: Time step size (s)
    - tau_rise, zeta_rise: Time constant and damping factor for the increasing phase
    - tau_fall, zeta_fall: Time constant and damping factor for the decreasing phase
    
    Returns:
    - C_measured: Smoothed concentration profile observed by the analyzer
    """
    Nt = len(C_model)
    C_measured = np.zeros(Nt)
    C_measured[0] = C_model[0]  # Initial condition
    C_dot = np.zeros(Nt)  # First derivative (rate of change)

    for t in range(1, Nt):
        # Check if the analyzer output is still increasing or has started decreasing
        if C_measured[t-1] >= C_measured[t-2]:  # Still rising
            tau, zeta = tau_rise, zeta_rise
        else:  # Transition to decreasing phase
            tau, zeta = tau_fall, zeta_fall

        # Compute second-order smoothing
        dC_dot_dt = (C_model[t] - C_measured[t-1] - 2*zeta*tau*C_dot[t-1]) / (tau**2)
        C_dot[t] = C_dot[t-1] + dC_dot_dt * dt
        C_measured[t] = C_measured[t-1] + C_dot[t] * dt

    return C_measured


def residuals(params, time_exp_CO2, concentration_exp_CO2,
              D, Time, Nx, Nt, epsilon, Omega, C_feed_ads_CO2, C_feed_ads_H2O,
              tau_rise, zeta_rise, tau_fall, zeta_fall):    
    
    k1, k2, k3 = params
    
    # Simulate the model
    try:
        C_CO2, C_H2O, theta_CO2_ads, theta_H2O_ads = simulate_adsorption_model(
            k1, k2, k3, T, P_ads, D, L, Time_ads, Nx, Nt, u, epsilon, 
            C_feed_ads_CO2, C_feed_ads_H2O, rho, Omega)
    except Exception as e:
        print(f"Simulation failed for parameters {params}: {e}")
        return np.full_like(concentration_exp_CO2, 1e6)
    
    # Model predictions at the end of the column
    C_CO2_ads_out = C_CO2[:, -1]
    # Convert CO2 and H2O outlet concentrations from mmol/mL to %
    C_CO2_percent = convert_mmol_per_mL_to_percent(C_CO2_ads_out, T, P_ads)
 
    # Apply analyzer delay to simulate measured values
    dt = Time / Nt
    C_measured_CO2 = apply_analyzer_delay_2nd_order(C_CO2_percent, dt, tau_rise, zeta_rise, tau_fall, zeta_fall)

    # Create a time array for the simulation output
    time_model = np.linspace(0, Time, len(C_measured_CO2))
    
    # Interpolate delayed simulation results to match experimental times
    concentration_model_interp_CO2 = np.interp(time_exp_CO2, time_model, C_measured_CO2)
    
    # Calculate residuals
    residuals_CO2 = concentration_model_interp_CO2 - concentration_exp_CO2

    return np.sum(residuals_CO2**2)


# start the computational time
start = perf_counter()


def objective(trial):
    # Variation Bounds
    k1 = trial.suggest_int('k1', 1, 200)
    k2 = trial.suggest_int('k2', 1, 200)
    k3 = trial.suggest_int('k3', 19, 20)

    params = [k1, k2, k3]

    # Call the existing residuals function
    res = residuals(
        params, 
        time_ads_exp_CO2, concentration_ads_exp_CO2,
        D, Time_ads, Nx, Nt_ads, epsilon, Omega, C_feed_ads_CO2, C_feed_ads_H2O,
        tau_rise, zeta_rise, tau_fall, zeta_fall
    )

    # Return sum of squares of residuals
    return 1e19 * np.sum(res**2)                

initial_guess = {
    'k1': 60,
    'k2': 20,
    'k3': 1
}


sampler = TPESampler(multivariate=True)

# Create the Optuna study
study = optuna.create_study(direction='minimize', sampler=sampler)

# Enqueue your guess as the first trial
study.enqueue_trial(initial_guess)

# Create study and optimize
study.optimize(objective, n_trials=1000, timeout=3600)  # 1 hour max

# Extract best parameters
best_params = study.best_params
print("\nBest parameters found by Optuna:")
for key, val in best_params.items():
    print(f"{key}: {val:.5f}")

k1_opt = best_params['k1']
k2_opt = best_params['k2']
k3_opt = best_params['k3']

duration = perf_counter() - start
print('{} took {:.3f} seconds\n\n'.format('Program', duration))

#simulate the model with the optimised parameters 
C_CO2_ads, C_H2O_ads, theta_CO2_ads, theta_H2O_ads = simulate_adsorption_model(k1_opt, k2_opt, k3_opt, T, P_ads, D, L, Time_ads, Nx, Nt_ads, u, epsilon, C_feed_ads_CO2, C_feed_ads_H2O, rho, Omega)

# Extract the outlet concentration over time from simulation
C_CO2_ads_out = C_CO2_ads[:, -1]     # CO2 concentration at the reactor outlet over time
C_H2O_ads_out = C_H2O_ads[:, -1]     # H2O concentration at the reactor outlet over time

Theta_CO2_ads_out = theta_CO2_ads[:, -1]           # CO2 coverage at the reactor outlet over time
Theta_H2O_ads_out = theta_H2O_ads[:, -1]           # H2O coverage at the reactor outlet over time

# Convert CO2 and H2O outlet concentrations from mmol/mL to %
C_CO2_percent = convert_mmol_per_mL_to_percent(C_CO2_ads_out, T, P_ads)
C_H2O_percent = convert_mmol_per_mL_to_percent(C_H2O_ads_out, T, P_ads)
# Time step size
dt = Time_ads / Nt_ads

# Apply the analyzer delay
C_CO2_measured = apply_analyzer_delay_2nd_order(C_CO2_percent, dt, tau_rise, zeta_rise, tau_fall, zeta_fall)
C_H2O_measured = apply_analyzer_delay_2nd_order(C_H2O_percent, dt, tau_rise, zeta_rise, tau_fall, zeta_fall)

# Time array
time = np.linspace(0, Time_ads, Nt_ads + 1)

# Plotting concentration at the end of the column versus time
plt.figure(figsize=(8, 4))
plt.plot(time, C_CO2_measured, label='CO2 Concentration at Column End', color='red')
plt.plot(time, C_CO2_percent, label='model output [CO2] at Column End', color='grey', linestyle='--')
plt.plot(time_ads_exp_CO2, concentration_ads_exp_CO2, 'o', label='CO2 Experimental Data', color='red', markersize=3, alpha=0.4)
plt.xlabel('Time (s)')
plt.ylabel('Concentration (vol.%)')
plt.title('Optimized CO2 Breakthrough Curve (Adsorption)')
param_text = (
    f"Best-fit parameters:\n"
    f"k1 = {k1_opt:.1f}, k2 = {k2_opt:.1f}, k3 = {k3_opt:.1f}, "
)

plt.figtext(0.5, -0.15, param_text, wrap=True, ha='center', fontsize=10)
plt.legend()
plt.grid(True)
plt.show()

# Plotting theta_CO2 and theta_H2O over time at the end of the column
plt.figure(figsize=(8, 4))
plt.plot(time, Theta_CO2_ads_out, label='Theta_CO2 at Column End', color='red')
plt.plot(time, Theta_H2O_ads_out, label='Theta_H2O at Column End', color='blue')
#plt.plot(time, Theta_CO2_H2O_ads_out, label='Theta_H2O/CO2 at Column End', color='black')
plt.xlabel('Time (s)')
plt.ylabel('Coverage Factor (Theta)')
plt.title('Coverage Factor (Theta) vs. Time at End of Column')
plt.legend()
plt.grid(True)
plt.show()

# Plotting H2O concentration at the end of the column versus time
plt.figure(figsize=(8, 4))
plt.plot(time, C_H2O_measured, label='H2O Concentration at Column End', color='blue')
plt.plot(time, C_H2O_percent, label='model output [H2O] at Column End', color='grey', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Concentration (vol.%)')
plt.title('H2O Concentration at the End of the Column Over Time')
plt.legend()
plt.grid(True)
plt.show()
