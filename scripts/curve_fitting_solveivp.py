import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Load the SIR data from the CSV file
sir_data = pd.read_csv('sir_data.csv')
I_observed = sir_data['Infected'].values
R_observed = sir_data['Recovered'].values
S_observed = sir_data['Susceptible'].values
population = S_observed[0] + I_observed[0] + R_observed[0]
days = len(sir_data)

# SIR model differential equations
def sir_model(t, y, beta, gamma):
    S, I, R = y
    dsdt = -beta * S * I / population
    didt = beta * S * I / population - gamma * I
    drdt = gamma * I
    return [dsdt, didt, drdt]

# Insert your estimated parameters here
beta_estimated = 0.29 #transmission rate of the disease higher value means more infectious
gamma_estimated = 0.095 #recovery rate of the disease higher value means faster recovery

# Initial conditions
y0 = [S_observed[0], I_observed[0], R_observed[0]]
t_span = [0, days]

# Integrate the SIR equations over the time grid using solve_ivp
solution = solve_ivp(sir_model, t_span, y0, args=(beta_estimated, gamma_estimated), method='DOP853', t_eval=np.arange(0, days))
S_fitted, I_fitted, R_fitted = solution.y

# Creating subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# Susceptible subplot
axs[0].plot(solution.t, S_observed, 'b.', alpha=0.5, label='Observed Susceptible')
axs[0].plot(solution.t, S_fitted, 'b-', label='Fitted Susceptible')
axs[0].set_ylabel('Susceptible')
axs[0].legend()

# Infected subplot
axs[1].plot(solution.t, I_observed, 'r.', alpha=0.5, label='Observed Infected')
axs[1].plot(solution.t, I_fitted, 'r-', label='Fitted Infected')
axs[1].set_ylabel('Infected')
axs[1].legend()

# Recovered subplot
axs[2].plot(solution.t, R_observed, 'g.', alpha=0.5, label='Observed Recovered')
axs[2].plot(solution.t, R_fitted, 'g-', label='Fitted Recovered')
axs[2].set_ylabel('Recovered')
axs[2].set_xlabel('Days')
axs[2].legend()

# Overall title and layout adjustment
plt.suptitle('SIR Model Curve Fitting with solve_ivp (DOP853)')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


beta_values = np.linspace(0.25, 0.3, 50)  # Adjust the range and number of values as needed
gamma_values = np.linspace(0.08, 0.11, 50)

utility_matrix = np.zeros((len(beta_values), len(gamma_values)))

for i, beta in enumerate(beta_values):
    for j, gamma in enumerate(gamma_values):
        solution = solve_ivp(sir_model, t_span, y0, args=(beta, gamma), method='DOP853', t_eval=np.arange(0, days))
        I_fitted = solution.y[1]
        utility = np.sum((I_observed - I_fitted) ** 2)
        utility_matrix[i, j] = utility


plt.figure(figsize=(10, 8))
plt.imshow(utility_matrix, extent=[beta_values.min(), beta_values.max(), gamma_values.min(), gamma_values.max()], origin='lower', cmap='viridis', aspect='auto')
plt.colorbar(label='Utility')
plt.xlabel('Beta')
plt.ylabel('Gamma')
plt.title('Utility Landscape')
plt.show()
