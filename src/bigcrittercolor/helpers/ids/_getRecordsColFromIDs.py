import pandas as pd

def _getRecordsColFromIDs(img_ids, column="species", data_folder=""):

    file_path = data_folder + "/records.csv"
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Filter the DataFrame to only include rows where the ID is in the set of ids
    filtered_df = df[df['img_id'].isin(img_ids)]

    # Create an ordered DataFrame based on the order of ids in the list
    filtered_df.set_index('img_id', inplace=True)
    ordered_df = filtered_df.loc[img_ids]

    # Return the values from the specified column
    return ordered_df[column].tolist()