import os
import pandas as pd

def _mergeAndSaveFolderCSVs(folder_path, save_path):
    # List all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Read and concatenate all CSV files
    dataframes = []
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        dataframes.append(pd.read_csv(file_path))

    # Combine all dataframes into a single dataframe
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Save the combined dataframe to the specified path
    combined_df.to_csv(save_path, index=False)
    print(f"Combined CSV saved to: {save_path}")