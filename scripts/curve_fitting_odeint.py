import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt

plt.style.use('IEEE_report')

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

# Insert your estimated parameters here
beta_estimated = 0.477
gamma_estimated = 0.3427

# Initial conditions
y0 = S_observed[0], I_observed[0], R_observed[0]
t = np.arange(0, days)

# Integrate the SIR equations over the time grid
solution = odeint(sir_model, y0, t, args=(beta_estimated, gamma_estimated))
S_fitted, I_fitted, R_fitted = solution.T

# Creating subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# Susceptible subplot
axs[0].plot(t, S_observed, 'b.', alpha=0.5, label='Observed Susceptible')
axs[0].plot(t, S_fitted, 'b-', label='Fitted Susceptible')
axs[0].set_ylabel('Susceptible')
axs[0].legend()

# Infected subplot
axs[1].plot(t, I_observed, 'r.', alpha=0.5, label='Observed Infected')
axs[1].plot(t, I_fitted, 'r-', label='Fitted Infected')
axs[1].set_ylabel('Infected')
axs[1].legend()

# Recovered subplot
axs[2].plot(t, R_observed, 'g.', alpha=0.5, label='Observed Recovered')
axs[2].plot(t, R_fitted, 'g-', label='Fitted Recovered')
axs[2].set_ylabel('Recovered')
axs[2].set_xlabel('Days')
axs[2].legend()

# Overall title and layout adjustment
plt.suptitle('SIR Model Curve Fitting')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
