""" PARAMETER ESTIMATION MODEL - HYDROGENATION """

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
import math as ma


# Parameters
Di = 2.216                      # Reactor (Tube) Inside Diameter (cm)   
Ai = 3.14 * (Di**2) / 4         # Reactor (Tube) Internal Surface Area (cm2)
L = 1                           # Length of the Reactor (tube) (cm)
W_DFM = 2                       # Weight of DFM in the reactor (gr) 
V_react = Ai * L                # Reactor (Tube) Volue (mL)
rho = W_DFM/V_react             # Adsorption bed Density (g/cm^3)

D = 0.16                    # Diffusion coefficient (cm^2/s)
epsilon = 0.35              # Porosity 0.35
R = 8.314e-3                # Ideal gas constant (J/K.mmol)
R_ = 0.08206                # (L.atm / mol.K)
P_ads = 1                   # Total pressure during adsorption (atm)
P_prg = 1                   # Total pressure during purge (atm)
P_hyd = 1                   # Total pressure during hydrogenation (atm)     

Time_ads = 1200             # Total simulation time (s) for adsorption 
Time_prg = 900              # Total simulation time (s) for purge 
Time_hyd = 1800             # Total simulation time (s) for hydrogenation 
Time_prg2 = 900             # Total simulation time (s) for the 2nd purge 
Nx = 5                      # Number of spatial grid points
Nt_ads = 3000               # Number of time steps
Nt_prg = 2000               # Number of time steps
Nt_hyd = 5000               # Number of time steps
Nt_prg2 = 2000              # Number of time steps

C_feed_ads_CO2 = 12.2       # CO2 %vol in feed gas - Adsorption stage (mmol/cm^3)   
C_feed_ads_H2O = 0          # H2O %vol in feed gas - Adsorption stage (mmol/cm^3)
C_feed_purge_CO2 = 0        # CO2 %vol in feed gas - Purge stage (mmol/cm^3)
C_feed_purge_H2O = 0        # H2O %vol in feed gas - Purge stage (mmol/cm^3)
C_feed_hyd_H2 = 10          # H2  %vol in feed gas - Hydrogenation stage (mmol/cm^3) 
C_feed_hyd_N2 = 100 - C_feed_hyd_H2        # N2  %vol in feed gas - Hydrogenation stage (mmol/cm^3)


# Fitted parameters from Adsorption Stage (for Water Ads/Des):
k2 = 12                     # Optimized kinetic constant for ads. of H2O on CO2 sites (cm3/s.g) 
k3 = 20                     # Optimized kinetic constant for ads. of H2O on H2O sites (cm3/s.g) 
n = 0.14                    # Adjust parameter in CH4 production 

# Analyzer delay response parameters Optimized:
tau_rise = 182.9
zeta_rise = 0.76
tau_fall = 222.4
zeta_fall = 0.78

# Simulate the purge process with the final state of the adsorption stage
C_CO2_init_prg = 12.2           # %C_CO2_adsorption[-1, :] 
C_H2O_init_prg = 0              # C_H2O_adsorption[-1, :]
theta_H2O_init_prg = 0          # theta_CO2_adsorption[-1, :]
theta_CO2_init_prg = 1          # theta_H2O_adsorption[-1, :]
theta_CO2_H2O_init_prg = 0      # theta_H2O_adsorption[-1, :]


# Load Methane Curves for all three temperatures (220 C, 300 C and 380 C)
CH4_380 = pd.read_excel('data/CH4-380.xlsx')
CH4_300 = pd.read_excel('data/CH4-300.xlsx')
CH4_220 = pd.read_excel('data/CH4-220.xlsx')

t_data_380 = CH4_380.iloc[:, 0].values
c_data_380 = CH4_380.iloc[:, 1].values

t_data_300 = CH4_300.iloc[:, 0].values
c_data_300 = CH4_300.iloc[:, 1].values

t_data_220 = CH4_220.iloc[:, 0].values
c_data_220 = CH4_220.iloc[:, 1].values

# Load H2O Curves for all three temperatures (220 C, 300 C and 380 C)
H2O_380 = pd.read_excel('data/H2O-380.xlsx')
H2O_300 = pd.read_excel('data/H2O-300.xlsx')
H2O_220 = pd.read_excel('data/H2O-220.xlsx')

t_data_380_H2O = H2O_380.iloc[:, 0].values
c_data_380_H2O = H2O_380.iloc[:, 1].values

t_data_300_H2O = H2O_300.iloc[:, 0].values
c_data_300_H2O = H2O_300.iloc[:, 1].values

t_data_220_H2O = H2O_220.iloc[:, 0].values
c_data_220_H2O = H2O_220.iloc[:, 1].values

# Load CO2 Curves for all three temperatures (220 C, 300 C and 380 C)
CO2_380 = pd.read_excel('data/CO2-380.xlsx')
CO2_300 = pd.read_excel('data/CO2-300.xlsx')
CO2_220 = pd.read_excel('data/CO2-220.xlsx')

t_data_380_CO2 = CO2_380.iloc[:, 0].values
c_data_380_CO2 = CO2_380.iloc[:, 1].values

t_data_300_CO2 = CO2_300.iloc[:, 0].values
c_data_300_CO2 = CO2_300.iloc[:, 1].values

t_data_220_CO2 = CO2_220.iloc[:, 0].values
c_data_220_CO2 = CO2_220.iloc[:, 1].values


def simulate_purge_model(k2, k3, k4, E4, Alfa,
                         T, P_prg, D, L, Time_prg, Nx, Nt, u, epsilon, 
                         C_CO2_feed_prg, C_H2O_feed_prg, C_CO2_init_prg, C_H2O_init_prg, 
                         theta_H2O_init_prg, rho, theta_CO2_init_prg, theta_CO2_H2O_init_prg, Omega):
    
    dx = L / (Nx - 1)
    dt = Time_prg / Nt
    x = np.linspace(0, L, Nx)
    C_CO2 = np.zeros((Nt + 1, Nx))
    C_H2O = np.zeros((Nt + 1, Nx))
    theta_CO2 = np.zeros((Nt + 1, Nx))
    theta_H2O = np.zeros((Nt + 1, Nx))
    Theta_CO2_H2O = np.zeros((Nt + 1, Nx))
    
    R_ = 0.08206               # (L.atm / mol.K)
    C_CO2_init = (C_CO2_feed_prg/100) * P_prg / (R_ * T)
    C_H2O_init = (C_H2O_feed_prg/100) * P_prg / (R_ * T)
    
    # Initial condition (linked to the previous stage)
    C_CO2[0, :] = (C_CO2_init_prg/100) * P_prg / (R_ * T)       # Initial concentration of CO2 in the column before purging
    C_H2O[0, :] = (C_H2O_init_prg/100) * P_prg / (R_ * T)       # Initial concentration of CO2 in the column before purging
    theta_H2O[0, :] = theta_H2O_init_prg                        # Initial coverage factor of H2O from adsorption stage
    theta_CO2[0, :] = theta_CO2_init_prg                        # Initial coverage factor of CO2 from adsorption stage
    Theta_CO2_H2O[0, :] = theta_CO2_H2O_init_prg                # Initial coverage factor of CO2/H2O joint adsorption
    
    def backward_euler_equations(y, i):
        C_CO2_next, C_H2O_next, theta_CO2_next, theta_H2O_next, Theta_CO2_H2O_next = y
        #C_CO2_next, C_H2O_next, theta_CO2_next, theta_H2O_next = y

        # Diffusion terms
        if i == Nx - 1:  # Outlet boundary condition
            diffusion_CO2_next = (D * dt / epsilon) * (-C_CO2_next + C_CO2[t, i - 1]) / dx**2
            diffusion_H2O_next = (D * dt / epsilon) * (-C_H2O_next + C_H2O[t, i - 1]) / dx**2
        elif i == 1:  # Inlet boundary condition
            diffusion_CO2_next = (D * dt / epsilon) * (C_CO2[t, i + 1] - C_CO2_next) / dx**2
            diffusion_H2O_next = (D * dt / epsilon) * (C_H2O[t, i + 1] - C_H2O_next) / dx**2
        else:  # Internal nodes
            diffusion_CO2_next = (D * dt / epsilon) * (C_CO2[t, i + 1] - 2 * C_CO2_next + C_CO2[t, i - 1]) / dx**2
            diffusion_H2O_next = (D * dt / epsilon) * (C_H2O[t, i + 1] - 2 * C_H2O_next + C_H2O[t, i - 1]) / dx**2

        # Convection terms
        convection_CO2_next = - (u * dt / epsilon) * (C_CO2_next - C_CO2[t, i - 1]) / dx
        convection_H2O_next = - (u * dt / epsilon) * (C_H2O_next - C_H2O[t, i - 1]) / dx

        # Reaction terms (The math contradiction (i.e., + instead of x) must be taken into account in future)
        CO2_formation_rate_next = k4 * np.exp((-E4 / (R * T)) * (1 - Alfa * theta_CO2_next)) * theta_CO2_next  # I've change "*" (as in the original paper) to "+" for multiplication of exponents
        H2O_formation_rate_next = k2 * C_CO2_next * theta_H2O_next - k3 * C_H2O_next * (1 - theta_CO2_next - theta_H2O_next)

        # Implicit equations
        eq1 = C_CO2_next - C_CO2[t, i] - diffusion_CO2_next - convection_CO2_next - (rho * CO2_formation_rate_next * dt / epsilon)
        eq2 = C_H2O_next - C_H2O[t, i] - diffusion_H2O_next - convection_H2O_next - (rho * H2O_formation_rate_next * dt / epsilon)
        eq3 = theta_CO2_next - theta_CO2[t, i] + (CO2_formation_rate_next * dt / Omega)
        eq4 = theta_H2O_next - theta_H2O[t, i] + (H2O_formation_rate_next * dt / Omega)
        eq5 = Theta_CO2_H2O_next - Theta_CO2_H2O[t, i] + ((CO2_formation_rate_next + H2O_formation_rate_next) * dt / Omega)

        return [eq1, eq2, eq3, eq4, eq5]
    for t in range(0,Nt):
        C_CO2[t + 1, 0] = C_CO2_init
        C_H2O[t + 1, 0] = C_H2O_init
        for i in range(1, Nx):
            initial_guess = [C_CO2[t, i], C_H2O[t, i], theta_CO2[t, i], theta_H2O[t, i], Theta_CO2_H2O[t, i]]
            solution = fsolve(backward_euler_equations, initial_guess, args=(i,))
            C_CO2[t + 1, i], C_H2O[t + 1, i], theta_CO2[t + 1, i], theta_H2O[t + 1, i], Theta_CO2_H2O[t + 1, i] = solution

        # Apply boundary conditions for inlet
        theta_CO2[t + 1, i] = np.clip(theta_CO2[t + 1, i], 0, 1)
        theta_H2O[t + 1, i] = np.clip(theta_H2O[t + 1, i], 0, 1)
        Theta_CO2_H2O[t + 1, i] = np.clip(Theta_CO2_H2O[t + 1, i], 0, 1)

    return C_CO2, C_H2O, theta_CO2, theta_H2O
        

# Function to simulate the model for hydrogenation of CO2
def simulate_hydrogenation_model(k6, k5, k8, k7, E6, E5, E7, R, T, P_hyd, n, D, L, Time_hyd, Nx, Nt, u, epsilon, rho, Omega, C_feed_hyd_H2, C_feed_hyd_N2, theta_CO2_hyd_initial):
    dx = L / (Nx - 1)
    dt = Time_hyd / Nt
    
    R_ = 0.08206               # (L.atm / mol.K)

    # Initialize concentration arrays
    C_CO2 = np.zeros((Nt + 1, Nx))
    C_H2 = np.zeros((Nt + 1, Nx))
    C_H2O = np.zeros((Nt + 1, Nx))
    C_CH4 = np.zeros((Nt + 1, Nx))
    C_N2 = np.zeros((Nt + 1, Nx))
    C_total = np.zeros((Nt + 1, Nx))
    theta_CO2 = np.zeros((Nt + 1, Nx))
    theta_H2O = np.zeros((Nt + 1, Nx))
    r_CH4_hyd_array = np.zeros((Nt + 1, Nx))
    r_CO2_hyd_array = np.zeros((Nt + 1, Nx))
    r_H2O_hyd_array = np.zeros((Nt + 1, Nx))

    # Initialize partial pressure arrays
    P_CO2 = np.zeros((Nt + 1, Nx))
    P_H2 = np.zeros((Nt + 1, Nx))
    P_CH4 = np.zeros((Nt + 1, Nx))
    P_H2O = np.zeros((Nt + 1, Nx))
    P_N2 = np.zeros((Nt + 1, Nx))
    
    # Initial conditions
    P_CO2[0, 0] = 0                                 # No CO2 at the beginning
    P_H2[0, 0] = (C_feed_hyd_H2/100) * P_hyd        # H2 is being fed at the inlet at t = 0
    P_CH4[0, 0] = 0                                 # No CH4 at the beginning
    P_H2O[0, 0] = 0                                 # No H2O at the beginning
    P_N2[0, 0] = (C_feed_hyd_N2/100) * P_hyd
    
    C_CO2[0, 0] = 0
    C_H2[0, 0] = P_H2[0, 0] / (R_ * T)              # mmol/mL
    C_H2O[0, 0] = 0
    C_CH4[0, 0] = 0
    C_N2[0, 0] = P_N2[0, 0] / (R_ * T)              # mmol/mL
    
    theta_CO2[0, :] = theta_CO2_hyd_initial         # Initial coverage factor from purge stage
    theta_H2O[0, :] = 0                             # Initial coverage factor from purge stage (will be changed in the cycle model)

    # Equilibrium constant for the reaction at given temperature
    K_eq = np.exp(0.5032 * ((56000 / T**2) + (34633 / T) - (16.4 * np.log(T)) + (0.00557 * T)) + 33.165)

    # Pre-calculate constants
    k6_exp = k6 * np.exp(-E6 / (R * T))
    k5_exp = k5 * np.exp(-E5 / (R * T))
    k7_exp = k7 * np.exp(-E7 / (R * T))

    def backward_euler_equations(y, i):
        C_CO2_next, C_H2_next, C_CH4_next, C_H2O_next, C_N2_next, theta_CO2_next, theta_H2O_next = y

        # Calculate total concentration at t+1
        C_total_next = P_hyd * 101325 / (R * T) / 1000000

        # Calculate partial pressures at t+1
        P_CO2_next = C_CO2_next / C_total_next * P_hyd
        P_H2_next = C_H2_next / C_total_next * P_hyd
        P_CH4_next = C_CH4_next / C_total_next * P_hyd
        P_H2O_next = C_H2O_next / C_total_next * P_hyd

        # Reaction rate calculations at time t+1
        # Rate of CO2 desorption
        r_CO2_ads_next = k6_exp * theta_CO2_next * C_H2_next
        
        # Rate of formation of CH4
        Approach_to_Equilibrium_next = P_CO2_next * (P_H2_next**4) - (P_CH4_next * P_H2O_next**2) / K_eq
        absApproach_to_Equilibrium_next = abs(Approach_to_Equilibrium_next)

        if absApproach_to_Equilibrium_next <= 1e-6:
            r_CH4_hyd_next = k5_exp * ma.copysign((268851.797358742 - (124307820284.15 * absApproach_to_Equilibrium_next)) * absApproach_to_Equilibrium_next, Approach_to_Equilibrium_next)
        else:
            r_CH4_hyd_next = k5_exp * ma.copysign(absApproach_to_Equilibrium_next**n, Approach_to_Equilibrium_next)

        r_H2O_ads_next = k7_exp * C_H2O_next * (1 - theta_CO2_next - theta_H2O_next) - k8 * theta_H2O_next

        # Diffusion terms with Dankwert boundary conditions at time t+1
        if i == Nx - 1:  # Outlet boundary (last node)
            diffusion_CO2_next = (D * dt / epsilon) * (-C_CO2_next + C_CO2[t, i - 1]) / dx**2
            diffusion_H2_next = (D * dt / epsilon) * (-C_H2_next + C_H2[t, i - 1]) / dx**2
            diffusion_CH4_next = (D * dt / epsilon) * (-C_CH4_next + C_CH4[t, i - 1]) / dx**2
            diffusion_H2O_next = (D * dt / epsilon) * (-C_H2O_next + C_H2O[t, i - 1]) / dx**2
            diffusion_N2_next = (D * dt / epsilon) * (-C_N2_next + C_N2[t, i - 1]) / dx**2

        elif i == 1:  # Inlet boundary (first internal node)
            diffusion_CO2_next = (D * dt / epsilon) * (C_CO2[t, i + 1] - C_CO2_next) / dx**2
            diffusion_H2_next = (D * dt / epsilon) * (C_H2[t, i + 1] - C_H2_next) / dx**2
            diffusion_CH4_next = (D * dt / epsilon) * (C_CH4[t, i + 1] - C_CH4_next) / dx**2
            diffusion_H2O_next = (D * dt / epsilon) * (C_H2O[t, i + 1] - C_H2O_next) / dx**2
            diffusion_N2_next = (D * dt / epsilon) * (C_N2[t, i + 1] - C_N2_next) / dx**2

        else:  # Internal nodes
            diffusion_CO2_next = (D * dt / epsilon) * (C_CO2[t, i + 1] - 2 * C_CO2_next + C_CO2[t, i - 1]) / dx**2
            diffusion_H2_next = (D * dt / epsilon) * (C_H2[t, i + 1] - 2 * C_H2_next + C_H2[t, i - 1]) / dx**2
            diffusion_CH4_next = (D * dt / epsilon) * (C_CH4[t, i + 1] - 2 * C_CH4_next + C_CH4[t, i - 1]) / dx**2
            diffusion_H2O_next = (D * dt / epsilon) * (C_H2O[t, i + 1] - 2 * C_H2O_next + C_H2O[t, i - 1]) / dx**2
            diffusion_N2_next = (D * dt / epsilon) * (C_N2[t, i + 1] - 2 * C_N2_next + C_N2[t, i - 1]) / dx**2

        # Convection terms (upwind scheme) at time t+1
        convection_CO2_next = - (u * dt / epsilon) * (C_CO2_next - C_CO2[t, i - 1]) / dx
        convection_H2_next = - (u * dt / epsilon) * (C_H2_next - C_H2[t, i - 1]) / dx
        convection_CH4_next = - (u * dt / epsilon) * (C_CH4_next - C_CH4[t, i - 1]) / dx
        convection_H2O_next = - (u * dt / epsilon) * (C_H2O_next - C_H2O[t, i - 1]) / dx
        convection_N2_next = - (u * dt / epsilon) * (C_N2_next - C_N2[t, i - 1]) / dx

        # Implicit Euler update
        eq1 = C_CO2_next - C_CO2[t, i] - diffusion_CO2_next - convection_CO2_next - (rho * (r_CO2_ads_next - r_CH4_hyd_next) * dt) / epsilon
        eq2 = C_CH4_next - C_CH4[t, i] - diffusion_CH4_next - convection_CH4_next - (rho * r_CH4_hyd_next * dt) / epsilon
        eq3 = C_H2_next - C_H2[t, i] - diffusion_H2_next - convection_H2_next - (rho * (-4 * r_CH4_hyd_next) * dt) / epsilon
        eq4 = C_H2O_next - C_H2O[t, i] - diffusion_H2O_next - convection_H2O_next - (rho * (-r_H2O_ads_next + 2 * r_CH4_hyd_next) * dt) / epsilon
        eq5 = C_N2_next - C_N2[t, i] - diffusion_N2_next - convection_N2_next

        # Update coverage factors to time t + 1
        # Desorption of CO2 from CO2-sites
        delta_theta_CO2_next = -r_CO2_ads_next * dt / Omega
        eq6 = theta_CO2_next - theta_CO2[t, i] - delta_theta_CO2_next

        delta_theta_H2O_next = (r_H2O_ads_next) * dt / Omega
        eq7 = theta_H2O_next - theta_H2O[t, i] - delta_theta_H2O_next

        return [eq1, eq2, eq3, eq4, eq5, eq6, eq7]  # Corrected return statement

    # Time loop
    for t in range(0, Nt):

        # Boundary conditions at the inlet (i = 0)
        C_CO2[t+1, 0] = C_CO2[t, 0]                      # No CO2 is being fed
        C_H2[t+1, 0] = C_feed_hyd_H2 / 100 * P_hyd / (R_ * T)  # H2 is being fed at the inlet
        C_CH4[t+1, 0] = C_CH4[t, 0]                     # No CH4 is being fed
        C_H2O[t+1, 0] = C_H2O[t, 0]                      # No H2O is being fed
        C_N2[t+1, 0] = C_feed_hyd_N2 / 100 * P_hyd / (R_ * T)  # N2 is being fed at the inlet

        # Spatial loop (using backward Euler)
        for i in range(1, Nx):
            # Solve the implicit equations
            initial_guess = [C_CO2[t, i], C_H2[t, i], C_CH4[t, i], C_H2O[t, i], C_N2[t, i], theta_CO2[t, i], theta_H2O[t, i]]
            solution = fsolve(backward_euler_equations, initial_guess, args=(i,))  # Pass initial_guess as y_prev

            # Unpack the solution
            C_CO2[t+1, i], C_H2[t+1, i], C_CH4[t+1, i], C_H2O[t+1, i], C_N2[t+1, i], theta_CO2[t+1, i], theta_H2O[t+1, i] = solution

        # Calculate total concentration at t+1
        C_total[t+1, :] = P_hyd * 101325 / (R * T) / 1000000  # Assuming constant total pressure

        # Calculate partial pressures at t+1
        P_CO2[t+1, :] = C_CO2[t+1, :] / C_total[t+1, :] * P_hyd
        P_H2[t+1, :] = C_H2[t+1, :] / C_total[t+1, :] * P_hyd
        P_CH4[t+1, :] = C_CH4[t+1, :] / C_total[t+1, :] * P_hyd
        P_H2O[t+1, :] = C_H2O[t+1, :] / C_total[t+1, :] * P_hyd
        P_N2[t+1, :] = C_N2[t+1, :] / C_total[t+1, :] * P_hyd

        # Calculate reaction rates and store them in arrays
        for i in range(1, Nx):
            r_CO2_ads = k6_exp * theta_CO2[t+1, i] * C_H2[t+1, i]

            Approach_to_Equilibrium = P_CO2[t+1, i] * (P_H2[t+1, i]**4) - (P_CH4[t+1, i] * P_H2O[t+1, i]**2) / K_eq
            absApproach_to_Equilibrium = abs(Approach_to_Equilibrium)

            if absApproach_to_Equilibrium <= 1e-6:
                r_CH4_hyd = k5_exp * ma.copysign((268851.797358742 - (124307820284.15 * absApproach_to_Equilibrium)) * absApproach_to_Equilibrium, Approach_to_Equilibrium)
            else:
                r_CH4_hyd = k5_exp * ma.copysign(absApproach_to_Equilibrium**n, Approach_to_Equilibrium)

            r_H2O_ads = k7_exp * C_H2O[t+1, i] * (1 - theta_CO2[t+1, i] - theta_H2O[t+1, i]) - k8 * theta_H2O[t+1, i]

            r_CH4_hyd_array[t+1, i] = r_CH4_hyd
            r_CO2_hyd_array[t+1, i] = r_CO2_ads  # Corrected to r_CO2_ads
            r_H2O_hyd_array[t+1, i] = r_H2O_ads

    return (C_CO2, C_CH4, C_H2, C_H2O, 
            theta_CO2, theta_H2O, 
            P_H2, P_CO2, P_CH4, P_H2O,
            r_CH4_hyd_array, r_CO2_hyd_array, r_H2O_hyd_array)


def convert_mmol_per_mL_to_percent(C_mmol_per_mL, T, P):
    R = 0.08206  # Gas constant in L·atm / mol·K
    C_percent = (C_mmol_per_mL * R * T / P) * 100
    return C_percent


def apply_analyzer_delay_2nd_order(C_model, dt, tau_rise, zeta_rise, tau_fall, zeta_fall):
    Nt = len(C_model)
    C_measured = np.zeros(Nt)
    C_measured[0] = C_model[0]  # Initial condition
    C_dot = np.zeros(Nt)  # First derivative (rate of change)

    for t in range(1, Nt):
        if C_measured[t-1] >= C_measured[t-2]:  # Still rising
            tau, zeta = tau_rise, zeta_rise
        else:  # Transition to decreasing phase
            tau, zeta = tau_fall, zeta_fall

        # Compute second-order smoothing
        dC_dot_dt = (C_model[t] - C_measured[t-1] - 2*zeta*tau*C_dot[t-1]) / (tau**2)
        C_dot[t] = C_dot[t-1] + dC_dot_dt * dt
        C_measured[t] = C_measured[t-1] + C_dot[t] * dt

    return C_measured


def residuals(params, T, u, Omega, theta_init, n,
              t_exp_CH4, c_exp_CH4,
              t_exp_H2O, c_exp_H2O,
              t_exp_CO2, c_exp_CO2,
              D, Time_hyd, Nx, Nt_hyd, epsilon,
              C_feed_hyd_H2, C_feed_hyd_N2,
              tau_rise, zeta_rise, tau_fall, zeta_fall):

    k6, E6, k5, E5, k8, k7, E7 = params

    try:
        C_CO2, C_CH4, _, C_H2O, *_ = simulate_hydrogenation_model(
            k6, k5, k8, k7, E6, E5, E7,
            R, T, P_hyd, n, D, L, Time_hyd, Nx, Nt_hyd, u,
            epsilon, rho, Omega, C_feed_hyd_H2, C_feed_hyd_N2, theta_init
        )
    except Exception as e:
        print(f"❌ Simulation failed at T={T}: {e}")
        return 1e6, 1e6, 1e6

    dt = Time_hyd / Nt_hyd
    time_model = np.linspace(0, Time_hyd, Nt_hyd + 1)

    try:
        CH4_pct = convert_mmol_per_mL_to_percent(C_CH4[:, -1], T, P_hyd)
        H2O_pct = convert_mmol_per_mL_to_percent(C_H2O[:, -1], T, P_hyd)
        CO2_pct = convert_mmol_per_mL_to_percent(C_CO2[:, -1], T, P_hyd)

        CH4_smooth = apply_analyzer_delay_2nd_order(CH4_pct, dt, tau_rise, zeta_rise, tau_fall, zeta_fall)
        H2O_smooth = apply_analyzer_delay_2nd_order(H2O_pct, dt, tau_rise, zeta_rise, tau_fall, zeta_fall)
        CO2_smooth = apply_analyzer_delay_2nd_order(CO2_pct, dt, tau_rise, zeta_rise, tau_fall, zeta_fall)

        CH4_model = np.interp(t_exp_CH4, time_model, CH4_smooth)
        H2O_model = np.interp(t_exp_H2O, time_model, H2O_smooth)
        CO2_model = np.interp(t_exp_CO2, time_model, CO2_smooth)
    except Exception as e:
        print(f"❌ Delay or interpolation failed at T={T}: {e}")
        return 1e6, 1e6, 1e6

    r_CH4 = np.sum((CH4_model - c_exp_CH4) ** 2)
    r_H2O = np.sum((H2O_model - c_exp_H2O) ** 2)
    r_CO2 = np.sum((CO2_model - c_exp_CO2) ** 2)

    return r_CH4, r_H2O, r_CO2


# start the computational time
start = perf_counter()


# Optuna objective function
def objective(trial):
    # === Hydrogenation parameters ===
    k6 = trial.suggest_int('k6', 5, 40)
    E6 = trial.suggest_int('E6', 5, 30)
    k5 = trial.suggest_int('k5', 42000, 58000)
    E5 = trial.suggest_int('E5', 65, 75)
    k8 = trial.suggest_float('k8', 0.001, 0.003, step=0.001)
    k7 = trial.suggest_int('k7', 140, 170)
    E7 = trial.suggest_int('E7', 8, 11)

    # === Purge parameters ===
    k4 = trial.suggest_float('k4', 0.001, 0.02, step=0.001)
    E4 = trial.suggest_int('E4', 15, 35)
    Alfa = trial.suggest_float('Alfa', 0.5, 0.9, step=0.1)

    # === Constants ===
    Temps = [653, 573, 493]
    u_vals = [1.95, 1.71, 1.47]
    Omega_vals = [0.448, 0.55, 0.665]
    theta_bounds = [(0.6, 0.72), (0.68, 0.82), (0.78, 0.95)]

    # Experimental data (per T)
    exp_CH4 = [(t_data_380, c_data_380), (t_data_300, c_data_300), (t_data_220, c_data_220)]
    exp_H2O = [(t_data_380_H2O, c_data_380_H2O), (t_data_300_H2O, c_data_300_H2O), (t_data_220_H2O, c_data_220_H2O)]
    exp_CO2 = [(t_data_380_CO2, c_data_380_CO2), (t_data_300_CO2, c_data_300_CO2), (t_data_220_CO2, c_data_220_CO2)]

    total_residual = 0

    for i in range(3):
        T = Temps[i]
        u = u_vals[i]
        Omega = Omega_vals[i]
        theta_min, theta_max = theta_bounds[i]

        # === Step 1: simulate purge ===
        try:
            _, _, theta_CO2_prg, *_ = simulate_purge_model(
                k2, k3, k4, E4, Alfa,
                T, P_prg, D, L,
                Time_prg, Nx, Nt_prg, u, epsilon,
                C_feed_purge_CO2, C_feed_purge_H2O,
                C_CO2_init_prg, C_H2O_init_prg,
                theta_H2O_init_prg, rho, theta_CO2_init_prg,
                theta_CO2_H2O_init_prg, Omega
            )
            theta_purge_outlet = theta_CO2_prg[-1, -1]
            print(f"✅ Purge θ_CO₂ at T={T} = {theta_purge_outlet:.4f}")
        except Exception as e:
            print(f"❌ Purge simulation failed at T={T}: {e}")
            return 1E4

        # === Step 2: enforce theta bounds ===
        if not (theta_min <= theta_purge_outlet <= theta_max):
            print(f"❌ Rejecting trial: purge theta={theta_purge_outlet:.3f} outside range {theta_min}-{theta_max} at T={T}")
            return 1e6

        # === Step 3: get corresponding experimental data ===
        t_CH4, c_CH4 = exp_CH4[i]
        t_H2O, c_H2O = exp_H2O[i]
        t_CO2, c_CO2 = exp_CO2[i]

        # === Step 4: calculate residuals (CH4, H2O, CO2 separately) ===
        try:
            r_CH4, r_H2O, r_CO2 = residuals(
                [k6, E6, k5, E5, k8, k7, E7],
                T, u, Omega, theta_purge_outlet, n,
                t_CH4, c_CH4, t_H2O, c_H2O, t_CO2, c_CO2,
                D, Time_hyd, Nx, Nt_hyd, epsilon,
                C_feed_hyd_H2, C_feed_hyd_N2,
                tau_rise, zeta_rise, tau_fall, zeta_fall
            )
        except Exception as e:
            print(f"❌ Hydrogenation residual failed at T={T}: {e}")
            return 1E4

        # === Step 5: apply weights ===
        w_CH4 = 1.0
        w_H2O = 0.8 if T != 493 else 0.5  # skip at 220°C
        w_CO2 = 0.5 if T != 493 else 0.2  # skip at 220°C

        total_residual += w_CH4 * r_CH4 + w_H2O * r_H2O + w_CO2 * r_CO2

    return total_residual


# Run the optimization
study = optuna.create_study(direction="minimize", sampler=TPESampler())
study.enqueue_trial({
    'k6': 35,
    'E6': 15,
    'k5': 58000,
    'E5': 65,
    'k8': 0.002,
    'k7': 154,
    'E7': 9,
    'k4': 0.003,
    'E4': 26,
    'Alfa': 0.6,
})


study.optimize(objective, n_trials=500, timeout=15000)

# Report best parameters
print("\nBest parameters found:")
for k, v in study.best_params.items():
    print(f"{k}: {v:.5f}")

best = study.best_params

# Extract optimised parameters
k6 = best['k6']
k5 = best['k5']
k8 = best['k8']
k7 = best['k7']
E6 = best['E6']
E5 = best['E5']
E7 = best['E7']
k4 = best['k4']
E4 = best['E4']
Alfa = best['Alfa']

# Temperature-specific parameters
Temps = [653, 573, 493]
u_vals = [1.95, 1.71, 1.47]
Omega_vals = [0.448, 0.55, 0.665]
labels = ['380°C', '300°C', '220°C']
colors = ['lightseagreen', 'purple', 'gold']

# Experimental data grouped by component
exp_data_CH4 = [(t_data_380, c_data_380), (t_data_300, c_data_300), (t_data_220, c_data_220)]
exp_data_H2O = [(t_data_380_H2O, c_data_380_H2O), (t_data_300_H2O, c_data_300_H2O), (t_data_220_H2O, c_data_220_H2O)]
exp_data_CO2 = [(t_data_380_CO2, c_data_380_CO2), (t_data_300_CO2, c_data_300_CO2), (t_data_220_CO2, c_data_220_CO2)]

# Lists to collect model curves and times
sim_times = []
model_curves_CH4 = []
model_curves_H2O = []
model_curves_CO2 = []

for i in range(3):
    T = Temps[i]
    u = u_vals[i]
    Omega = Omega_vals[i]


    # Simulate purge to get theta_init
    _, _, theta_CO2_prg, *_ = simulate_purge_model(
        k2, k3, k4, E4, Alfa,
        T, P_prg, D, L, Time_prg, Nx, Nt_prg, u, epsilon,
        C_feed_purge_CO2, C_feed_purge_H2O,
        C_CO2_init_prg, C_H2O_init_prg,
        theta_H2O_init_prg, rho, theta_CO2_init_prg,
        theta_CO2_H2O_init_prg, Omega
    )
    theta_init = theta_CO2_prg[-1, -1]

    # Simulate hydrogenation
    C_CO2, C_CH4, _, C_H2O, *_ = simulate_hydrogenation_model(
        k6, k5, k8, k7, E6, E5, E7, R, T, P_hyd, n, D, L,
        Time_hyd, Nx, Nt_hyd, u, epsilon, rho, Omega,
        C_feed_hyd_H2, C_feed_hyd_N2, theta_init
    )

    dt = Time_hyd / Nt_hyd
    sim_time = np.linspace(0, Time_hyd, Nt_hyd + 1)
    sim_times.append(sim_time)

    # Convert to % and apply analyzer delay
    CH4_pct = convert_mmol_per_mL_to_percent(C_CH4[:, -1], T, P_hyd)
    H2O_pct = convert_mmol_per_mL_to_percent(C_H2O[:, -1], T, P_hyd)
    CO2_pct = convert_mmol_per_mL_to_percent(C_CO2[:, -1], T, P_hyd)

    CH4_smooth = apply_analyzer_delay_2nd_order(CH4_pct, dt, tau_rise, zeta_rise, tau_fall, zeta_fall)
    H2O_smooth = apply_analyzer_delay_2nd_order(H2O_pct, dt, tau_rise, zeta_rise, tau_fall, zeta_fall)
    CO2_smooth = apply_analyzer_delay_2nd_order(CO2_pct, dt, tau_rise, zeta_rise, tau_fall, zeta_fall)

    model_curves_CH4.append(CH4_smooth)
    model_curves_H2O.append(H2O_smooth)
    model_curves_CO2.append(CO2_smooth)

# --- Report θ_CO₂ values from purge for each temperature ---
print("\nFinal θ_CO₂ values from Purge stage (theta_init) after optimisation:")

theta_final_dict = {}  # to collect and optionally reuse later

for i, T in enumerate(Temps):
    u = u_vals[i]
    Omega = Omega_vals[i]

    _, _, theta_CO2_prg, *_ = simulate_purge_model(
        k2, k3, k4, E4, Alfa,
        T, P_prg, D, L, Time_prg, Nx, Nt_prg, u, epsilon,
        C_feed_purge_CO2, C_feed_purge_H2O,
        C_CO2_init_prg, C_H2O_init_prg,
        theta_H2O_init_prg, rho, theta_CO2_init_prg,
        theta_CO2_H2O_init_prg, Omega
    )

    theta_init = theta_CO2_prg[-1, -1]
    theta_final_dict[T] = theta_init
    print(f"T = {T} K → θ_CO₂ = {theta_init:.4f}")


# === CH4 Plot ===
plt.figure(figsize=(8, 4))
for i in range(3):
    t_exp, c_exp = exp_data_CH4[i]
    plt.plot(sim_times[i], model_curves_CH4[i], label=f'Model {labels[i]}', color=colors[i])
    plt.plot(t_exp, c_exp, 'o', label=f'Exp {labels[i]}', color=colors[i], markersize=4, alpha=0.4)
plt.title("Methane (CH₄) Outlet Concentration")
plt.xlabel("Time (s)")
plt.ylabel("CH₄ [% vol]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === H2O Plot ===
plt.figure(figsize=(8, 4))
for i in range(3):
    t_exp, c_exp = exp_data_H2O[i]
    plt.plot(sim_times[i], model_curves_H2O[i], label=f'Model {labels[i]}', color=colors[i])
    plt.plot(t_exp, c_exp, 'o', label=f'Exp {labels[i]}', color=colors[i], markersize=4, alpha=0.4)
plt.title("Water (H₂O) Outlet Concentration")
plt.xlabel("Time (s)")
plt.ylabel("H₂O [% vol]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === CO2 Plot ===
plt.figure(figsize=(8, 4))
for i in range(3):
    t_exp, c_exp = exp_data_CO2[i]
    plt.plot(sim_times[i], model_curves_CO2[i], label=f'Model {labels[i]}', color=colors[i])
    plt.plot(t_exp, c_exp, 'o', label=f'Exp {labels[i]}', color=colors[i], markersize=4, alpha=0.4)
plt.title("Carbon Dioxide (CO₂) Outlet Concentration")
plt.xlabel("Time (s)")
plt.ylabel("CO₂ [% vol]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


duration = perf_counter() - start
print('{} took {:.3f} seconds\n\n'.format('Program', duration))
