import os

from bigcrittercolor.helpers import _getBCCIDs, _bprint

# stage can be "image", "mask", "segment"
def readSpeciesBalancedIDs(stage_type="image",print_steps=True,data_folder=''):
    _bprint(print_steps, "Reading previously saved species-balanced ids...")
    # Read the balanced ids from the text file
    balanced_ids_path = os.path.join(data_folder, "species_balanced_ids.txt")
    with open(balanced_ids_path, 'r') as f:
        balanced_ids = [line.strip() for line in f.readlines()]

    if stage_type is "image":
        return balanced_ids

    _bprint(print_steps, "Filtering for species-balanced ids that have reached the {stage_type} stage...")
    # stage is either "mask" or "segment"
    stage_ids = _getBCCIDs(type=stage_type,data_folder=data_folder)

    # get balanced ids that have actually reached the stage
    balanced_ids = set(balanced_ids).intersection(set(stage_ids))

    return balanced_ids