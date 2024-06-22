import pandas as pd
import os

# Define file paths
raw_data_path = "data/raw/wdbc.data"
processed_data_path = "data/processed/wdbc_processed.csv"

# Define column names based on the dataset description
column_names = ["ID", "Diagnosis",
                "Radius Mean", "Texture Mean", "Perimeter Mean", "Area Mean", "Smoothness Mean",
                "Compactness Mean", "Concavity Mean", "Concave Points Mean", "Symmetry Mean", "Fractal Dimension Mean",
                "Radius SE", "Texture SE", "Perimeter SE", "Area SE", "Smoothness SE",
                "Compactness SE", "Concavity SE", "Concave Points SE", "Symmetry SE", "Fractal Dimension SE",
                "Radius Worst", "Texture Worst", "Perimeter Worst", "Area Worst", "Smoothness Worst",
                "Compactness Worst", "Concavity Worst", "Concave Points Worst", "Symmetry Worst", "Fractal Dimension Worst"]

# Load the data
data = pd.read_csv(raw_data_path, header=None, names=column_names)

# Drop the ID column as it is not useful for analysis
data.drop(columns=["ID"], inplace=True)

# Encode the Diagnosis column (B -> 0, M -> 1)
data['Diagnosis'] = data['Diagnosis'].map({'B': 0, 'M': 1})

# Save the processed data to a new file
os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
data.to_csv(processed_data_path, index=False)

print("Data preprocessing complete. Processed data saved to:", processed_data_path)
