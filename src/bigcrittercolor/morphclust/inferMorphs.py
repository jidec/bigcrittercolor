import pandas as pd
import cv2
import copy

from bigcrittercolor.helpers import _clusterByImgFeatures, _bprint, _getBCCIDs, _showImages, _readBCCImgs
from bigcrittercolor.helpers.ids import _getRecordsColFromIDs
from bigcrittercolor.helpers.verticalize import _verticalizeImg
from bigcrittercolor.helpers.image import _equalize
from bigcrittercolor.helpers import _clusterByImgHistograms

# this fun should:
# 1. group records by species
# 2. load in segments (already filtered using filterExtractSegs)
# 3. cluster them on resnet features and depending on some threshold figure out if they have morphs
# 4. bind the morph info as a new column records (usable as a group clustering column for example)

def inferMorphs(img_ids=None, group_records_colname="species",strategy="histogram",
                cluster_params_dict={'algo': "dbscan", 'tsne_n':2,'show_pca_tsne':True,'show_silhouette':True},
                print_steps=True, print_details=False, show=True, data_folder=""):

    # histogram, features, bioencodings
    # if no ids get all existing
    if img_ids is None:
        _bprint(print_steps, "No IDs specified, getting all IDs from existing segments...")
        img_ids = _getBCCIDs(type="segment", data_folder=data_folder)

    # read in segs
    segs = _readBCCImgs(img_ids=img_ids, type="segment", data_folder=data_folder)

    # equalize
    segs = [_equalize(seg, type="histogram") for seg in segs]

    # get species names matching ids
    species = _getRecordsColFromIDs(img_ids=img_ids, column="species", data_folder=data_folder)

    # init species morph list (will become a column later) with same length as species
    all_species_morphs = copy.deepcopy(species)

    # for each unique species
    species_uniq = list(set(species))
    for sp in species_uniq:
        # get indices for that species
        indices = [i for i, x in enumerate(species) if x == sp]
        # use indices to get segs and matching ids for that species
        segs_sp = [segs[i] for i in indices]
        ids_sp = [img_ids[i] for i in indices]

        # continue (skip) if n obs is less than threshold - note that species morph column is just the same as sp in this case
        if len(ids_sp) < 20:
            continue

        # cluster the segs
        # labels = _clusterByImgFeatures(segs_sp, cluster_params_dict=cluster_params_dict,show=True)
        labels = _clusterByImgHistograms(segs_sp, cluster_params_dict)
        #labels = _clusterByBioEncoderFeatures

        # add new morph labels to their indices
        for index, (species_name, label) in zip(indices, zip(species, labels)):
            all_species_morphs[index] = f"{species_name}_{label}"
        # all_species_morphs[indices] = [o[0] + "_" + str(o[1]) for o in zip(species,labels)]

    # make df of morph labels and ids
    df = pd.DataFrame({'img_id': img_ids, 'species_morph': all_species_morphs})

    # read in records
    # records = pd.read_csv(data_folder + "/records.csv")

    # merge
    # records = pd.merge(records, df, on='img_id')

    # write

#inferMorphs(data_folder="D:/anac_tests")