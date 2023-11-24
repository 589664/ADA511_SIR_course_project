import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize

# Load the SIR data from the CSV file
sir_data = pd.read_csv('sir_data.csv')
I_observed = sir_data['Infected'].values
R_observed = sir_data['Recovered'].values
S_observed = sir_data['Susceptible'].values
population = S_observed[0] + I_observed[0] + R_observed[0]
days = len(sir_data)

# SIR model differential equations
def sir_model(y, t, beta, gamma):
    S, I, R = y
    dsdt = -beta * S * I / population
    didt = beta * S * I / population - gamma * I
    drdt = gamma * I
    return dsdt, didt, drdt

# Cost function to minimize
def sir_cost(params):
    beta, gamma = params
    y0 = S_observed[0], I_observed[0], R_observed[0]
    t = np.arange(0, days)
    solution = odeint(sir_model, y0, t, args=(beta, gamma))
    S, I, R = solution.T
    return np.mean((I_observed - I)**2 + (R_observed - R)**2)

# Initial guess for the parameters
initial_guess = [0.3, 0.1]

# Perform the optimization
optimal_params = minimize(sir_cost, initial_guess, method='L-BFGS-B', bounds=[(0, 1), (0, 1)])

beta_estimated, gamma_estimated = optimal_params.x

print(f"Estimated Beta: {beta_estimated}")
print(f"Estimated Gamma: {gamma_estimated}")
