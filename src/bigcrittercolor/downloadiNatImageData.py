import pandas as pd
import os
import time
from datetime import datetime
import shutil
from pathlib import Path
import copy
import cv2

from bigcrittercolor.helpers import _bprint, _inat_dl_helpers, _rebuildiNatRecords, _writeBCCImgs, _getBCCIDs
from bigcrittercolor.helpers.ids import _imgNameToID

# works by:
# 1. calls getiNatRecords which loads prev obs ids
# 2. then it calls getRecords check if we have downloaded records already - if we have, load those and skip
# 3. getRecords also get the rec count and splits it up...
# 4. it calls retrieveRecords which downloads but skips records already in the downloaded RECORDS (not images)...

# notably:
# 1. images that are already downloaded are skipped
# 2. a new copy of records is always downloaded, so new observations since the last time we ran the fun will always be downloaded
# 3. you should keep lat long box the same when downloading new obs or things will get screwed up
# 4. interrupting it in the middle will not cause any issues
# 5. if we're removing already downloaded images from records, it's BAD if the fun gets interrupted and the original
#   isn't restored because we need the original when binding everything together into the main supp df
# 6. however,
# 7. skip_existing_taxa bypasses downloading a new copy of records, and assumes that the records already downloaded are complete - this
#   will fail to download all records for a taxon if a previous records download was interrupted
# 8. works in two pieces - downloading records and downloading images using records
def downloadiNatImageData(taxa_list, download_records=True, download_images=True,
                          lat_lon_box=None, usa_only=False, img_size="medium", research_grade_only=True, #n_per_taxon=None, max_n_per_obs=1,
                          skip_records_for_existing_taxa=False,
                          print_steps=True, data_folder=''):
    """ Download iNat images and data by genus or species.
        This method downloads records then refers to those records to download in parallel.
        Images are always saved in data_folder/all_images.
        Based on Brian Stucky's iNat downloader, his modified code is in _inat_dl_helpers.py.

        Args:
           taxa_list (list): the list of genera and/or species. List members are treated as species if binomial (i.e. "Homo sapiens") and genus if not (i.e. "Homo")
           lat_lon_box (tuple): tuple of the form ((sw_lat, sw_lon), (ne_lat,ne_lon))
           skip_existing (bool): if we skip images that have already been downloaded
           img_size (str): size of images to download - can be "medium", "large", or "full"
           data_folder (str): the path to the bigcrittercolor formatted data folder
    """

    # if usa only set the lat lon box
    if usa_only and lat_lon_box is None:
        lat_lon_box = ((24.396308,-124.848974),(49.384358,-66.885444))

    taxa_list_records = copy.deepcopy(taxa_list)
    if skip_records_for_existing_taxa:
        # List to hold the filenames
        filenames_starting_with_iNat = []
        path = data_folder + "/other/inat_download_records"
        # Iterate over the entries in the directory
        for entry in os.listdir(path):
            # Check if the entry is a file and if its name starts with "iNat"
            if os.path.isfile(os.path.join(path, entry)) and entry.startswith("iNat"):
                filenames_starting_with_iNat.append(entry)
        taxa_with_records = [s.split('-')[1].split('.')[0] for s in filenames_starting_with_iNat]
        taxa_list_records = [item for item in taxa_list if item not in taxa_with_records]

    # for every taxon
    if download_records:
        for taxon in taxa_list_records:
            _bprint(print_steps,"Retrieving records for taxon " + taxon + "...")
            _inat_dl_helpers.getiNatRecords(taxon=taxon,research_grade_only=research_grade_only,lat_lon_box=lat_lon_box,img_size=img_size,data_folder=data_folder)

    if download_images:
        # n_per_taxon does not work as expected currently
        #if n_per_taxon is not None:
            # read in records for taxon
        #    records = pd.read_csv(data_folder + '/other/inat_download_records/iNat_images-' + taxon + '.csv')
        #    full_taxon_records = records.copy()

            # sample
        #    if len(records) > n_per_taxon:
        #        records = records.sample(n_per_taxon)
            # replace records with new trimmed records
        #    records.to_csv(data_folder + '/other/inat_download_records/iNat_images-' + taxon + '.csv', index=False,
        #                   mode='w+')
        for taxon in taxa_list:
            _bprint(print_steps, "Starting image download for taxon " + taxon + "...")

            # create the dir for split records
            split_dir = data_folder + '/other/inat_download_records/iNat_images-' + taxon + '-records_split'
            if not os.path.isdir(split_dir):
                os.mkdir(split_dir)
            # create the dir for trimmed records
            dir = data_folder + '/other/inat_download_records/iNat_images-' + taxon + '-records_trimmed'
            if not os.path.isdir(dir):
                os.mkdir(dir)

            # get only records NOT in existing records for taxon
            # get existing ids
            existing_ids = _getBCCIDs(type="image", data_folder=data_folder)
            existing_imgnames = [id + ".jpg" for id in existing_ids] # convert to existing imgnames
            existing_ogfilenames = [i.replace('INAT-', '') for i in existing_imgnames] # remove INAT- prefix to get original filenames

            # read in records for taxon
            records = pd.read_csv(data_folder + '/other/inat_download_records/iNat_images-' + taxon + '.csv')
            # get only records NOT in existing records for taxon
            in_mask = records["file_name"].isin(existing_ogfilenames)
            records = records[~in_mask]
            # remove gifs (WEIRD that these are in here)
            records = records[~records['img_url'].str.contains('.gif')]
            n_new_obs = records.shape[0]
            if n_new_obs == 0:
                _bprint(print_steps, "No new observations to download, exiting or moving to next taxon...")
                continue
            _bprint(print_steps, "Will download " + str(n_new_obs) + " new observations...")

            records_path = data_folder + '/other/inat_download_records/iNat_images-' + taxon + '-records_trimmed/trimmed_records.csv'

            # write trimmed records to trimmed records folder
            records.to_csv(
                records_path,
                index=False, mode='w+')

            _bprint(print_steps, "Using records to download images...")
            # make raw images folder if it doesn't exist
            rawimg_folder = data_folder + "/other/inat_download_records/iNat_images-" + taxon + "-raw_images"
            if not os.path.exists(rawimg_folder):
                os.mkdir(rawimg_folder)

            fileout = data_folder + "/other/inat_download_records/" + taxon + '-download_log.csv'
            # download the images
            _inat_dl_helpers.downloadImages(img_records=records_path,imgdir=rawimg_folder,fileout=fileout)

            # rename raw images with INAT- prefix
            imgnames = os.listdir(rawimg_folder)
            for name in imgnames:
                newname = name.split('_')[0]
                os.rename(rawimg_folder + "/" + name,rawimg_folder + "/INAT-" + newname)

            _bprint(print_steps, "Moving images to dataset...")
            # rename images as JPGs and move
            filenames = os.listdir(rawimg_folder)
            paths = [rawimg_folder + "/" + filename for filename in filenames]
            imgs = [cv2.imread(path) for path in paths]
            imgnames = [filename + ".jpg" for filename in filenames]

            # write the images
            _writeBCCImgs(imgs,imgnames,data_folder=data_folder)

            # remove the raw images dir
            shutil.rmtree(rawimg_folder)

    _rebuildiNatRecords(data_folder=data_folder)
    _bprint(print_steps, "Finished.")

