import os

# Define the folder containing the CSV files
folder_path = './covid_data'

# List all files in the folder
files = os.listdir(folder_path)

# Loop through the files and delete those with a 2020 date
for filename in files:
    if filename.endswith('.csv'):
        parts = filename.split('-')
        if len(parts) == 3 and parts[2].startswith('2020'):
            file_path = os.path.join(folder_path, filename)
            os.remove(file_path)
