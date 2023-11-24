import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time

# Load data
data = pd.read_csv('combined_sir_data_brazil.csv')

# Total population
N = 212600000

# Extract data columns
S = data['Susceptible']
I = data['Confirmed']
R = data['Recovered']
T = data['Day']

# Define the SIR model equations (discrete form)
def sir_model(t, y, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# Define the objective function to minimize (sum of squared differences)
def objective(params):
    beta, gamma = params
    y0 = [S0, I0, R0]  # Initial conditions
    solution = solve_ivp(
        lambda t, y: sir_model(t, y, beta, gamma), [T.iloc[0], T.iloc[-1]], y0, t_eval=T
    )
    y_pred = solution.y[1]  # Predicted number of infected cases
    return np.sum((y_pred - I.values) ** 2)

# Initial conditions
I0 = data.iloc[0]['Confirmed']
R0 = data.iloc[0]['Recovered']
S0 = data.iloc[0]['Susceptible']

# Initial guess for parameters
initial_guess = [0.2, 0.1]

# Set a maximum time for parameter estimation (in seconds)
max_time = 300  # 5 minutes

# Define a callback function to check elapsed time
start_time = time.time()
def callback(params):
    elapsed_time = time.time() - start_time
    if elapsed_time > max_time:
        raise Exception("Parameter estimation exceeded the time limit")

# Perform parameter estimation using the least squares method with a time limit
result = minimize(objective, initial_guess, callback=callback)
estimated_params = result.x

# Extract estimated beta and gamma
estimated_beta, estimated_gamma = estimated_params

# Print the estimated parameters
print(f"Estimated beta: {estimated_beta:.4f}")
print(f"Estimated gamma: {estimated_gamma:.4f}")
