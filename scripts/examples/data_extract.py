import os
import pandas as pd

# Define the folder containing the CSV files
folder_path = './covid_data'

# Initialize an empty DataFrame to store the combined data
combined_data = pd.DataFrame()

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        # Read the CSV file into a DataFrame
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        
        # Filter the data for Norway
        norway_data = df[df['Country_Region'] == 'Denmark'].copy()
        
        # Set the 'Date' column in the copied DataFrame
        norway_data['Date'] = pd.to_datetime(norway_data['Last_Update']).dt.date
        
        # Select only the relevant columns
        relevant_columns = norway_data[['Date', 'Confirmed', 'Deaths', 'Recovered', 'Active']]
        
        # Append the relevant data to the combined_data DataFrame
        combined_data = pd.concat([combined_data, relevant_columns])

# Sort the combined data by date in ascending order
combined_data = combined_data.sort_values(by='Date', ascending=True)

# Save the combined data to a single CSV file
combined_data.to_csv('norway_covid_data.csv', index=False)
