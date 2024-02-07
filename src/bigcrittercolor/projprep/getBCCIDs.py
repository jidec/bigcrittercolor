import pandas as pd
from bigcrittercolor.helpers import _getIDsInFolder

def getBCCIDs(taxa_list,
              has_mask=False, has_segment=False, has_pattern=True,
              data_folder=''):

    # Load the CSV file into a pandas DataFrame
    data = pd.read_csv(data_folder + "/records.csv")

    # Initialize an empty list to store matching obs_ids
    matched_obs_ids = []

    for taxon in taxa_list:
        # Determine if the taxon is a species (contains spaces) or a genus (no spaces)
        if ' ' in taxon:
            # It's a species
            filtered_data = data[data['species'] == taxon]
        else:
            # It's a genus
            filtered_data = data[data['genus'] == taxon]

        # Append the obs_ids from the filtered data to the list
        matched_obs_ids.extend(filtered_data['obs_id'].tolist())

    if has_mask:
        mask_ids = _getIDsInFolder(data_folder + "/masks")
        matched_obs_ids = list(set(matched_obs_ids) & set(mask_ids))
    if has_segment:
        seg_ids = _getIDsInFolder(data_folder + "/segments")
        matched_obs_ids = list(set(matched_obs_ids) & set(seg_ids))
    if has_segment:
        pat_ids = _getIDsInFolder(data_folder + "/patterns")
        matched_obs_ids = list(set(matched_obs_ids) & set(pat_ids))

    # Remove duplicates by converting the list to a set and back to a list
    matched_obs_ids = list(set(matched_obs_ids))

    return matched_obs_ids