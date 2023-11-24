import pandas as pd

# Assuming the population of Brazil at the time of the earliest timestamp (1/22/20)
population_of_brazil = 212600000  # You would need to replace this with the correct value

# Read the datasets
confirmed_df = pd.read_csv('confirmed.csv')
deaths_df = pd.read_csv('deaths.csv')
recovered_df = pd.read_csv('recovered.csv')

# Filter for Brazil and the first 250 days
confirmed_brazil = confirmed_df[confirmed_df['Country/Region'] == 'Brazil'].iloc[:, 54:304]
deaths_brazil = deaths_df[deaths_df['Country/Region'] == 'Brazil'].iloc[:, 54:304]
recovered_brazil = recovered_df[recovered_df['Country/Region'] == 'Brazil'].iloc[:, 54:304]

# confirmed_brazil = confirmed_df[confirmed_df['Country/Region'] == 'Brazil'].iloc[:, 4:254]
# deaths_brazil = deaths_df[deaths_df['Country/Region'] == 'Brazil'].iloc[:, 4:254]
# recovered_brazil = recovered_df[recovered_df['Country/Region'] == 'Brazil'].iloc[:, 4:254]

# Calculate the susceptible component
susceptible = population_of_brazil - (confirmed_brazil + recovered_brazil)

# Create a list of day numbers from 1 to 250
days = list(range(1, 251))

# Create a new DataFrame with the specified columns
combined_df = pd.DataFrame({
    'Day': days,
    'Susceptible': susceptible.values.flatten(),
    'Confirmed': confirmed_brazil.values.flatten(),
    'Recovered': recovered_brazil.values.flatten(),
    'Deaths': deaths_brazil.values.flatten()
})

# Save the combined data to a new CSV file
combined_df.to_csv('combined_sir_data_brazil.csv', index=False)
