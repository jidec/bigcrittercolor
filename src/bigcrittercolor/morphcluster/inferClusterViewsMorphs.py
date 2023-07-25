import pandas as pd
import cv2
from bigcrittercolor.helpers import _clusterByImgFeatures, _bprint, _getIDsInFolder, showIDs, _extractVerticalizeMasksOrSegs

def inferClusterViewsMorphs(img_ids, records_group_col="species",
                       print_steps=True, print_details=False, show=True, data_folder=""):
    # if no ids get all existing
    if img_ids is None:
        _bprint(print_steps, "No IDs specified, getting all IDs from existing segments...")
        img_ids = _getIDsInFolder(data_folder + "/segments")

    # get locations of segs
    seg_locs = [data_folder + "/segments/" + id + "_segment.jpg" for id in img_ids]

    # get locations of masks - all segs WILL have matching masks if not something has gone wrong earlier
    mask_locs = [data_folder + "/masks/" + id + "_mask.jpg" for id in img_ids]

    print(mask_locs)
    # read in masks
    masks = []
    mask_ids = []
    # for each image location
    for index, mask_loc in enumerate(mask_locs):
        mask = cv2.imread(mask_loc) #cv2.IMREAD_GRAYSCALE)
        masks.append(mask)
        mask_ids.append(img_ids[index])

    print(len(masks))

    # normalize masks
    masks = _extractVerticalizeMasksOrSegs(masks)

    # cluster the masks and visualize
    labels = _clusterByImgFeatures(masks, show=True)

    # read in records
    records = pd.read_csv(data_folder + "/records.csv")

    # make list of tuples where each tuple contains a mask id and its cluster label
    id_label_tuples_list = list(zip(mask_ids, labels))

    # merge a new DataFrame created with the tuple list with the existing DataFrame based on the 'ID' column
    records = pd.merge(records, pd.DataFrame(id_label_tuples_list, columns=['img_id', 'view_cluster']), on='img_id', how='left')

    # get unique groups (species or genera) in records
    groups = pd.unique(records.loc[:, records_group_col])

    # get unique views
    views = pd.unique(records.loc[:, 'view_cluster'])

    # for each species
    for g in groups:
        # for each view
        for v in views:
            # get records of group and view
            recs = records.loc[(records[records_group_col] == g) & (records['view_cluster'] == v)]
            # get ids
            recs_ids = recs['recordID']
            _bprint(print_details, "Got records for group " + g + " and view label " + str(v))

            if show:
                showIDs(recs_ids,data_folder=data_folder)





