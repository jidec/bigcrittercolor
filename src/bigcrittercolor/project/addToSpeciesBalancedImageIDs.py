import os
import random

from bigcrittercolor.helpers import _getBCCIDs, _bprint
from bigcrittercolor.helpers.ids import _getSpeciesBalancedIDs
from bigcrittercolor.project import readSpeciesBalancedIDs
from bigcrittercolor.helpers.ids import _getSpeciesBalancedIDs

def addToSpeciesBalancedImageIDs(data_folder='', n_new_ids=50):

    existing_ids = readSpeciesBalancedIDs(data_folder=data_folder)

    new_ids = _getSpeciesBalancedIDs(type="image", n_per_species=n_new_ids,
                                     print_steps=False, print_details=False,
                                     data_folder=data_folder)

    # Combine existing ids with the new ones
    combined_ids = list(set(existing_ids + new_ids))  # Ensure no duplicates# Save the new combined ids to a new text file
    new_balanced_ids_path = os.path.join(data_folder, "species_balanced_ids_updated.txt")
    with open(new_balanced_ids_path, 'w') as f:
        for img_id in combined_ids:
            f.write(f"{img_id}\n")

    _bprint(True, f"Added {len(new_ids)} new IDs. Total IDs now: {len(combined_ids)}")