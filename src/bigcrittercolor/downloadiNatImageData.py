import pandas as pd
import os
import time
from datetime import datetime
import shutil
from pathlib import Path

from bigcrittercolor.helpers import _bprint, _inat_dl_helpers

def downloadiNatImageData(taxa_list, lat_lon_box=None, usa_only=False, img_size="medium",
                          skip_existing=True, print_steps=True,
                          n_per_taxon=None,
                          data_folder='../..'):
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

    # for every taxon
    for taxon in taxa_list:
        _bprint(print_steps,"Retrieving records for taxon " + taxon + "...")
        _inat_dl_helpers.getiNatRecords(taxon=taxon,research_only=True,lat_lon_box=lat_lon_box,img_size=img_size,data_folder=data_folder)

        # n_per_taxon does not work as expected currently
        if n_per_taxon is not None:
            # read in records for taxon
            records = pd.read_csv(data_folder + '/other/inat_download_records/iNat_images-' + taxon + '.csv')
            full_taxon_records = records.copy()

            # sample
            if len(records) > n_per_taxon:
                records = records.sample(n_per_taxon)
            # replace records with new trimmed records
            records.to_csv(data_folder + '/other/inat_download_records/iNat_images-' + taxon + '.csv', index=False,
                           mode='w+')

        # if skipping existing (not redownloading), remove already downloaded images from the records
        if skip_existing:
            _bprint(print_steps, "Skipping existing already downloaded images...")
            # get existing image names
            existing_imgnames = os.listdir(data_folder + '/all_images')

            # remove INAT- prefix to get original filenames
            existing_ogfilenames = [i.replace('INAT-', '') for i in existing_imgnames]

            # read in records for taxon
            records = pd.read_csv(data_folder + '/other/inat_download_records/iNat_images-' + taxon + '.csv')
            # save these records
            full_taxon_records = records.copy()

            # get only records NOT in existing records for taxon
            in_mask = records["file_name"].isin(existing_ogfilenames)
            records = records[~in_mask]
            _bprint(print_steps, "Will download " + str(records.shape[0]) + " new observations...")

            # replace records with new trimmed records
            records.to_csv(data_folder + '/other/inat_download_records/iNat_images-' + taxon + '.csv',index=False,mode='w+')

        # create the dir for split records if it doesn't exist
        split_dir = data_folder + '/other/inat_download_records/iNat_images-' + taxon + '-records_split'
        if not os.path.isdir(split_dir):
            os.mkdir(split_dir)

        _bprint(print_steps, "Splitting records into chunks...")
        # split the records using pd.read_csv with the chunksize arg
        j = 1
        for chunk in pd.read_csv(data_folder + '/other/inat_download_records/iNat_images-' + taxon + '.csv', chunksize=7500):
            chunk.to_csv(split_dir + '/' + str(j) + '.csv', index=False)
            j += 1

        _bprint(print_steps, "Using records to download images for taxon " + taxon + "...")
        # make raw images folder if it doesn't exist
        rawimg_dir = data_folder + "/other/inat_download_records/iNat_images-" + taxon + "-raw_images"
        if not os.path.exists(rawimg_dir):
            os.mkdir(rawimg_dir)

        dirname = data_folder + '/other/inat_download_records/iNat_images-' + taxon + "-raw_images"
        # download each record chunk, waiting an hour between
        for c, record_chunk in enumerate(os.listdir(data_folder + '/other/inat_download_records/iNat_images-' + taxon + '-records_split')):
            fileout = data_folder + "/other/inat_download_records/" + taxon + '-download_log.csv'
            # download the images
            _inat_dl_helpers.downloadImages(img_records=data_folder + '/other/inat_download_records/iNat_images-' + taxon + '.csv',imgdir=dirname,fileout=fileout)

            # add INAT- prefix to images
            imgnames = os.listdir(rawimg_dir)
            for name in imgnames:
                newname = name.split('_')[0]
                os.rename(dirname + "/" + name,dirname + "/INAT-" + newname)

            _bprint(print_steps, "Renaming images as JPGs and moving them to all_images...")
            # rename images as JPGs and move
            dirname = data_folder + '/other/inat_download_records/iNat_images-' + taxon + "-raw_images"
            for i, filename in enumerate(os.listdir(dirname)):
                file = Path(dirname + "/" + filename + ".jpg")
                if not file.exists():
                    os.rename(dirname + "/" + filename, dirname + "/" + filename + ".jpg")
                shutil.move(dirname + "/" + filename + ".jpg", data_folder + "/all_images/" + filename + ".jpg")

            # TODO could allow changing chunksize since medium images hit the servers more weakly
            # if there are more chunks left, wait for an hour before doing the next chunk
            if c < len(os.listdir(data_folder + '/other/inat_download_records/iNat_images-' + taxon + '-records_split')) - 1:
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print("Waiting one hour starting at " + current_time + "...")
                time.sleep(3600)

        # write full records csv back
        full_taxon_records.to_csv(data_folder + '/other/inat_download_records/iNat_images-' + taxon + '.csv', index=False, mode='w+')

    _bprint(print_steps, "Finished.")

