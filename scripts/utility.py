import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Load the SIR data from the CSV file
sir_data = pd.read_csv('./data/sir_data.csv')
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

# Estimated parameters
beta_estimated = 0.29 #transmission rate of the disease higher value means more infectious
gamma_estimated = 0.1 #recovery rate of the disease higher value means faster recovery

# Initial conditions
y0 = [S_observed[0], I_observed[0], R_observed[0]]
t_span = [0, days]

# Integrate the SIR equations over the time grid using solve_ivp
solution = solve_ivp(sir_model, t_span, y0, args=(beta_estimated, gamma_estimated), method='DOP853', t_eval=np.arange(0, days))
S_fitted, I_fitted, R_fitted = solution.y

# Create a range of beta and gamma values
beta_values = np.linspace(0.27, 0.29, 30)
gamma_values = np.linspace(0.08, 0.1, 50)

# print(beta_values)
# print(gamma_values)

# Initialize arrays to store utilities
utility_beta = np.zeros(len(beta_values))
utility_gamma = np.zeros(len(gamma_values))

# Calculate utilities for different beta values
for i, beta in enumerate(beta_values):
    solution = solve_ivp(sir_model, t_span, y0, args=(beta, gamma_estimated), method='DOP853', t_eval=np.arange(0, days))
    I_fitted = solution.y[1]
    utility = np.sum((I_observed - I_fitted) ** 2)
    utility_beta[i] = utility

# Calculate utilities for different gamma values
for i, gamma in enumerate(gamma_values):
    solution = solve_ivp(sir_model, t_span, y0, args=(beta_estimated, gamma), method='DOP853', t_eval=np.arange(0, days))
    I_fitted = solution.y[1]
    utility = np.sum((I_observed - I_fitted) ** 2)
    utility_gamma[i] = utility

# # Create two histograms
# plt.figure(figsize=(12, 5))

# # Histogram for beta
# plt.subplot(1, 2, 1)
# plt.hist(utility_beta, bins=20, color='blue', alpha=0.7)
# plt.xlabel('Utility (Beta)')
# plt.ylabel('Frequency')
# plt.title('Histogram of Utility for Beta')

# # Histogram for gamma
# plt.subplot(1, 2, 2)
# plt.hist(utility_gamma, bins=20, color='green', alpha=0.7)
# plt.xlabel('Utility (Gamma)')
# plt.ylabel('Frequency')
# plt.title('Histogram of Utility for Gamma')

# plt.tight_layout()
# plt.show()


def utility_function(observed, fitted, compartment):
    # Define the penalties
    underestimation_penalty = 1.5  # Assuming underestimation is twice as bad as overestimation
    overestimation_penalty = 1

    # Calculate the error array
    error = observed - fitted

    # Initialize the utility array
    utility = np.zeros(observed.shape)

    # Apply different penalties for underestimation and overestimation for each element
    for i in range(len(error)):
        if compartment == 'Infected':
            if error[i] > 0:
                # Underestimation of Infected is worse
                utility[i] = -underestimation_penalty * (error[i] ** 2)
            else:
                # Overestimation of Infected is less severe
                utility[i] = -overestimation_penalty * (error[i] ** 2)
        elif compartment == 'Susceptible':
            # Symmetric penalty for Susceptible
            utility[i] = -(error[i] ** 2)
        elif compartment == 'Recovered':
            if error[i] > 0:
                # Overestimation of Recovered might give a false sense of security
                utility[i] = -overestimation_penalty * (error[i] ** 2)
            else:
                # Underestimation of Recovered might be seen as being cautious
                utility[i] = -underestimation_penalty * (error[i] ** 2)
        else:
            raise ValueError("Invalid compartment name")

    return utility.sum()  # Return the sum of all utilities for the entire array


# Example usage for beta values
for i, beta in enumerate(beta_values):
    solution = solve_ivp(sir_model, t_span, y0, args=(beta, gamma_estimated), method='DOP853', t_eval=np.arange(0, days))
    S_fitted, I_fitted, R_fitted = solution.y
    utility_S = utility_function(S_observed, S_fitted, 'Susceptible')
    utility_I = utility_function(I_observed, I_fitted, 'Infected')
    utility_R = utility_function(R_observed, R_fitted, 'Recovered')
    combined_utility = utility_S + utility_I + utility_R
    utility_beta[i] = combined_utility

# Make sure the utilities have been calculated using the loop from the previous messages

# Now let's plot the utility values against beta and gamma values
plt.figure(figsize=(14, 7))

# Plot for beta
plt.subplot(1, 2, 1)
plt.plot(beta_values, utility_beta, marker='o', linestyle='-', color='blue')
plt.title('Combined Utility for Varying Beta')
plt.xlabel('Beta Values')
plt.ylabel('Combined Utility')

# Plot for gamma
plt.subplot(1, 2, 2)
plt.plot(gamma_values, utility_gamma, marker='o', linestyle='-', color='green')
plt.title('Combined Utility for Varying Gamma')
plt.xlabel('Gamma Values')
plt.ylabel('Combined Utility')

plt.tight_layout()
plt.show()
