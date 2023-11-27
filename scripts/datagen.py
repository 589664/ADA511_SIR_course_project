import numpy as np
import pandas as pd

# Parameters for the SIR model
population = 10000000
initial_infected = 100
initial_recovered = 0
days = 200

def sir_model(s, i, r, beta, gamma):
    dsdt = -beta * s * i / population
    didt = beta * s * i / population - gamma * i
    drdt = gamma * i
    return dsdt, didt, drdt

t = np.linspace(0, days, days)
S = np.zeros(days)
I = np.zeros(days)
R = np.zeros(days)

S[0] = population - initial_infected - initial_recovered
I[0] = initial_infected
R[0] = initial_recovered

for day in range(1, days):
    # Base values for beta and gamma
    beta = 0.3
    gamma = 0.1

    # Introduce variations at specific intervals
    if day % 45 == 0:  # Every quarter of the year
        beta *= 3  # Increase infection rate by 30%
        gamma *= 0.8  # Decrease recovery rate by 10%

    dsdt, didt, drdt = sir_model(S[day - 1], I[day - 1], R[day - 1], beta, gamma)

    S[day] = S[day - 1] + dsdt
    I[day] = I[day - 1] + didt
    R[day] = R[day - 1] + drdt

    I[day] = max(I[day], 0)
    R[day] = max(R[day], 0)

sir_data = pd.DataFrame({
    'Day': t.astype(int),
    'Susceptible': S.astype(int),
    'Infected': I.astype(int),
    'Recovered': R.astype(int)
})

sir_data.to_csv('./data/sir_data.csv', index=False)
print(sir_data.head())
