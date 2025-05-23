import pandas as pd
import random
import os

from bigcrittercolor.helpers import _getBCCIDs, _bprint

def _getSpeciesBalancedIDs(type="image", n_per_species=200, usa_only=False, print_steps=True, print_details=True,
                           data_folder=''):
    # Bounding box: ((min_lat, min_lon), (max_lat, max_lon))
    lat_lon_box = ((24.396308, -124.848974), (49.384358, -66.885444))

    _bprint._bprint(print_steps, "Started _getSpeciesBalancedIDs...")

    csv_path = os.path.join(data_folder, "records.csv")
    data = pd.read_csv(csv_path)

    _bprint._bprint(print_steps, "Getting all downloaded image ids...")
    downloaded_ids = _getBCCIDs._getBCCIDs(type=type, data_folder=data_folder)

    filtered_data = data[data['img_id'].isin(downloaded_ids)]

    # Filter by US bounding box if requested
    if usa_only:
        (min_lat, min_lon), (max_lat, max_lon) = lat_lon_box
        filtered_data = filtered_data[
            (filtered_data['latitude'] >= min_lat) & (filtered_data['latitude'] <= max_lat) &
            (filtered_data['longitude'] >= min_lon) & (filtered_data['longitude'] <= max_lon)
            ]
        _bprint._bprint(print_steps, f"Filtered to {len(filtered_data)} USA-only records.")

    _bprint._bprint(print_steps, "Sampling IDs for each species...")
    balanced_ids = []
    species_count = {}
    for species, group in filtered_data.groupby('species'):
        sampled_ids = group['img_id'].sample(min(n_per_species, len(group)), random_state=1).tolist()
        balanced_ids.extend(sampled_ids)
        species_count[species] = len(sampled_ids)

    if print_details:
        for species, count in species_count.items():
            print(f"{species}: {count} ids")

    _bprint._bprint(print_steps, "Finished _getSpeciesBalancedIDs.")
    return balanced_ids
