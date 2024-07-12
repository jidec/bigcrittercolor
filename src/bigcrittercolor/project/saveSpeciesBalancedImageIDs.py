import pandas as pd
import random
import os

from bigcrittercolor.helpers import _getBCCIDs, _bprint
from bigcrittercolor.helpers.ids import _getSpeciesBalancedIDs

def saveSpeciesBalancedImageIDs(n_per_species=200, print_steps=True, print_details=True,data_folder=''):

    balanced_ids = _getSpeciesBalancedIDs(type="image",n_per_species=n_per_species,
                                          print_steps=print_steps,print_details=print_details,
                                          data_folder=data_folder)
    _bprint(print_steps, "Writing species-balanced ids...")
    # write the balanced ids to a text file
    balanced_ids_path = os.path.join(data_folder, "species_balanced_ids.txt")
    with open(balanced_ids_path, 'w') as f:
        for img_id in balanced_ids:
            f.write(f"{img_id}\n")