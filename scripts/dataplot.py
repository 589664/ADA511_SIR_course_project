import pandas as pd
import matplotlib.pyplot as plt

# Replace 'your_data.csv' with the path to your CSV file
data = pd.read_csv('sir_data.csv')

# Assuming the population of Brazil at the time of the earliest timestamp (1/22/20)
population = 10000000  

# Calculate percentages
data['Susceptible'] = (data['Susceptible'] / population) * 100
data['Infected'] = (data['Infected'] / population) * 100
data['Recovered'] = (data['Recovered'] / population) * 100

# Plotting
plt.figure(figsize=(12, 8))

# Plot Susceptible
plt.subplot(1, 2, 1)
plt.plot(data['Day'], data['Susceptible'], label='Actual Susceptible')

# Plot Infected and Removed
plt.subplot(1, 2, 2)
plt.plot(data['Day'], data['Infected'], label='Actual Infected')
plt.plot(data['Day'], data['Recovered'], label='Actual Removed')

# Add legends and labels
plt.subplot(1, 2, 1)
plt.xlabel('Time (Days)')
plt.ylabel('Percent of Population')
plt.legend()

plt.subplot(1, 2, 2)
plt.xlabel('Time (Days)')
plt.ylabel('Percent of Population')
plt.legend()

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()
