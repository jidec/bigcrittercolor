import os
import pandas as pd

def _rebuildiNatRecords(data_folder):
    # get a list of all files in the folder
    files = os.listdir(data_folder + "/other/inat_download_records")

    # filter the files based on the specified string
    csv_filenames = [file for file in files if file.endswith('.csv') and "iNat_images-" in file]

    # initialize an empty DataFrame to store the merged data
    merged_data = pd.DataFrame()

    # iterate over the CSV files and merge them
    for csv_filename in csv_filenames:
        file_path = data_folder + "/other/inat_download_records/" + csv_filename
        try:
            data = pd.read_csv(file_path)
            # Skip empty dataframes
            if not data.empty:
                merged_data = pd.concat([merged_data, data])
        except pd.errors.EmptyDataError:
            print(f"Warning: {file_path} is empty and will be skipped.")

    # keep useful columns
    merged_data = merged_data[['obs_id', 'taxon', 'latitude', 'longitude', 'annotations',
                               'img_url', 'img_cnt','date']]

    # rename taxon col to species
    merged_data = merged_data.rename(columns={'taxon': 'species'})
    # create genus col by spltting species col on spaces
    merged_data['genus'] = merged_data['species'].str.split(' ', n=1).str[0]
    # add INAT to the obs ids
    merged_data['obs_id'] = 'INAT-' + merged_data['obs_id'].astype(str)
    merged_data['img_id'] = merged_data['obs_id'].astype(str) + "-1"

    # write
    merged_data.to_csv(data_folder + "/records.csv", index=False)

#_rebuildiNatRecords("D:/anac_tests")