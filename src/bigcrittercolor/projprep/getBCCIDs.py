import pandas as pd
from bigcrittercolor.helpers import _getIDsInFolder

def getBCCIDs(taxa_list=None,
              has_mask=False, has_segment=False, has_pattern=False,
              data_folder=''):

    # start with all the image ids (of form INAT-XY-Z)
    img_ids = _getIDsInFolder(data_folder + "/all_images")

    if taxa_list is not None:
        # load records containing taxonomic and other info
        records = pd.read_csv(data_folder + "/records.csv")

        # list to store all the record ids for the taxa (of form INAT-XY)
        taxa_record_ids = []
        for taxon in taxa_list:
            # determine if the taxon is a species (contains spaces) or a genus (no spaces)
            if ' ' in taxon:
                # It's a species
                filtered_data = records[records['species'] == taxon]
            else:
                # It's a genus
                filtered_data = records[records['genus'] == taxon]

            # append the obs_ids from the filtered data to the list
            taxa_record_ids.extend(filtered_data['obs_id'].tolist())


        # splitting each id in detailed_ids at "-" and removing the last part to compare
        img_ids = [id for id in img_ids if "-".join(id.split("-")[:-1]) in taxa_record_ids]

    if has_mask:
        mask_ids = _getIDsInFolder(data_folder + "/masks")
        img_ids = list(set(img_ids) & set(mask_ids))
    if has_segment:
        seg_ids = _getIDsInFolder(data_folder + "/segments")
        img_ids = list(set(img_ids) & set(seg_ids))
    if has_pattern:
        pat_ids = _getIDsInFolder(data_folder + "/patterns")
        img_ids = list(set(img_ids) & set(pat_ids))

    # remove duplicates by converting the list to a set and back to a list
    img_ids = list(set(img_ids))

    return img_ids