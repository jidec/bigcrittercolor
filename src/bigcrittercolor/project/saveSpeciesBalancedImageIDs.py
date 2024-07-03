import pandas as pd
import random
import os

from bigcrittercolor.helpers import _getBCCIDs, _bprint

def saveSpeciesBalancedImageIDs(n_per_species=200, print_steps=True, print_details=True,data_folder=''):
    # read the CSV file
    csv_path = os.path.join(data_folder, "records.csv")
    data = pd.read_csv(csv_path)

    _bprint(print_steps, "Getting all downloaded image ids...")
    # get the list of ids that we actually have downloaded
    downloaded_ids = _getBCCIDs(type="image",data_folder=data_folder)

    # filter the data for rows that have ids contained
    filtered_data = data[data['img_id'].isin(downloaded_ids)]

    _bprint(print_steps, "Sampling IDs for each species...")
    # group by species and sample ids
    balanced_ids = []
    species_count = {}
    for species, group in filtered_data.groupby('species'):
        sampled_ids = group['img_id'].sample(min(n_per_species, len(group)), random_state=1).tolist()
        balanced_ids.extend(sampled_ids)
        species_count[species] = len(sampled_ids)

    if print_details:
        # Print the number of species and how many ids were obtained for each
        for species, count in species_count.items():
            print(f"{species}: {count} ids")

    _bprint(print_steps, "Writing species-balanced ids...")
    # write the balanced ids to a text file
    balanced_ids_path = os.path.join(data_folder, "species_balanced_ids.txt")
    with open(balanced_ids_path, 'w') as f:
        for img_id in balanced_ids:
            f.write(f"{img_id}\n")