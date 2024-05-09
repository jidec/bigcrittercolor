import pandas as pd
from bigcrittercolor.helpers import _getIDsInFolder

def getBCCIDs2(taxa_list=None,
              has_mask=False, has_segment=False, has_pattern=False,
              data_folder=''):
    # Load all image ids
    img_ids = set(_getIDsInFolder(data_folder + "/all_images"))

    # Filtering by taxa
    if taxa_list is not None:
        records = pd.read_csv(data_folder + "/records.csv")

        # Determine whether each taxon is species or genus
        taxa_list = [(taxon, ' ' in taxon) for taxon in taxa_list]

        # Vectorized filtering using Pandas queries
        species = [taxon[0] for taxon in taxa_list if taxon[1]]
        genera = [taxon[0] for taxon in taxa_list if not taxon[1]]

        species_filtered = records[records['species'].isin(species)]['obs_id'].tolist() if species else []
        genera_filtered = records[records['genus'].isin(genera)]['obs_id'].tolist() if genera else []

        taxa_record_ids = set(species_filtered + genera_filtered)

        # Apply taxa filtering directly to img_ids
        img_ids = {id for id in img_ids if "-".join(id.split("-")[:-1]) in taxa_record_ids}

    # Helper function to apply mask/segment/pattern filters
    def apply_filter(folder_name):
        return set(_getIDsInFolder(data_folder + f"/{folder_name}"))

    # Apply additional filters
    if has_mask:
        mask_ids = apply_filter("masks")
        img_ids &= mask_ids
    if has_segment:
        seg_ids = apply_filter("segments")
        img_ids &= seg_ids
    if has_pattern:
        pat_ids = apply_filter("patterns")
        img_ids &= pat_ids

    return list(img_ids)