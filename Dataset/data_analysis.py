import pandas as pd
import os

# Directory containing your CSV files
data_dir = "/path/to/your/csv/directory"

# List all files in the directory
all_files = os.listdir(data_dir)

# Filter for only CSV files
csv_files = [f for f in all_files if f.endswith('.csv')]

# Loop through each CSV file and process it
for csv_file in csv_files:
    # Construct the full file path
    file_path = os.path.join(data_dir, csv_file)
    
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Perform operations on the DataFrame, for example:
    print(f"Processing {csv_file}")
    print(df.head())  # Display the first few rows

    # Add any further analysis or processing steps as needed

print("All files processed.")
