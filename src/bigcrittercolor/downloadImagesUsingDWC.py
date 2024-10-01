import os
import pandas as pd
import cv2
import shutil

from bigcrittercolor.helpers import _bprint, _inat_dl_helpers, _rebuildiNatRecords, _writeBCCImgs, _getBCCIDs

def downloadImagesUsingDWC(inat_img_size="medium", max_n_per_obs=1,
                          download_n_per_species=None,
                          skip_existing=True,
                          print_steps=True, data_folder=''):

    # setup paths
    records_csv_path = os.path.join(data_folder, "records.csv")
    darwincore_folder_path = os.path.join(data_folder, "other", "gbif_darwincore_records_split")
    # set chunk size for downloading
    download_chunk_size = 5000
    # only create records if either of the paths does not exist
    if not os.path.exists(records_csv_path) or not os.path.exists(darwincore_folder_path):
        _bprint(print_steps, "Creating records for image download using DarwinCore in other/my_gbif_darwincore...")

        # get name of DarwinCore folder as the only folder in other/my_gbif_darwincore
        other_folder = os.path.join(data_folder, "other", "my_gbif_darwincore")
        folder_names = [name for name in os.listdir(other_folder) if os.path.isdir(os.path.join(other_folder, name))]
        if len(folder_names) == 0:
            raise FileNotFoundError("No DarwinCore archive folder found in other/my_gbif_darwincore, did you add it?")
        target_folder = os.path.join(other_folder, folder_names[0])  # Choose the first folder

        # define the paths to the TSV files
        multimedia_path = os.path.join(target_folder, "multimedia.txt")
        occurrence_path = os.path.join(target_folder, "occurrence.txt")

        # load the TSV files into pandas DataFrames
        multimedia_df = pd.read_csv(multimedia_path, sep='\t')
        occurrence_df = pd.read_csv(occurrence_path, sep='\t')

        # drop duplicates in multimedia_df keeping only the first match for each gbifID
        multimedia_df = multimedia_df.drop_duplicates(subset='gbifID', keep='first')

        # merge the two dataframes on gbifID
        merged_df = pd.merge(occurrence_df, multimedia_df, on='gbifID', how='left')
        merged_df["imageLink"] = merged_df["identifier"]
        merged_df["rightsHolder"] = merged_df['rightsHolder_x']

        # keep only relevant columns
        merged_df = merged_df[["gbifID","datasetName","informationWithheld",
        "occurrenceID","catalogNumber","sex","lifeStage","caste","occurrenceRemarks",
        "eventDate","eventTime","startDayOfYear","endDayOfYear","year","month","day","verbatimEventDate",
        "decimalLatitude","decimalLongitude","coordinateUncertaintyInMeters",
        "identificationID", "identifiedBy",
        "scientificName","kingdom","phylum","class","order","superfamily","family","genus","infraspecificEpithet","taxonomicStatus",
        "imageLink","rightsHolder"]]

        merged_df["imageLink"] = merged_df["imageLink"].str.replace("original", "medium", regex=False)
        # create img_url and file_name cols required by downloader
        merged_df["img_url"] = merged_df["imageLink"]
        merged_df["file_name"] = merged_df["catalogNumber"]

        # keep only rows with image urls
        merged_df = merged_df.loc[merged_df['img_url'].notna() & (merged_df['img_url'] != "")]

        # write to records in root
        merged_df.to_csv(data_folder + "/records.csv", index=False)

        # split records into .csvs which will be passed to the downloader (download chunks)
        splitCSVToFolder(data_folder + "/records.csv", download_chunk_size, output_folder=data_folder + "/other/split_gbif_download_records")

    # location of download log
    fileout = data_folder + "/other/download_log.csv"
    # location of split .csvs
    csv_folder = data_folder + "/other/split_gbif_download_records"
    # get a list of all CSV files in the directory
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

    # loop through each .csv file and download all the images in each
    for csv_file in csv_files:
        records_path = os.path.join(csv_folder, csv_file)
        print(f"Processing {csv_file}...")

        temp_raw_img_folder = data_folder + "/other/temp_raw_imgs"
        os.mkdir(temp_raw_img_folder)

        # download the images
        _inat_dl_helpers.downloadImages(img_records=records_path, imgdir=temp_raw_img_folder, fileout=fileout)

        # if records are iNaturalist...
        img_name_prefix = "INAT"

        # rename raw images with prefix
        imgnames = os.listdir(temp_raw_img_folder)
        for name in imgnames:
            newname = name.split('_')[0]
            os.rename(temp_raw_img_folder + "/" + name, temp_raw_img_folder + "/" + img_name_prefix + "-" + newname)

        _bprint(print_steps, "Moving images to dataset...")

        # get all raw image filenames, paths, images, imgnames
        filenames = os.listdir(temp_raw_img_folder)
        paths = [os.path.join(temp_raw_img_folder, filename) for filename in filenames]
        imgs = [cv2.imread(path) for path in paths]
        imgnames = [filename + ".jpg" for filename in filenames]
        # write to the dataset
        _writeBCCImgs(imgs, imgnames, data_folder=data_folder)

        # remove the raw images dir when moving is done
        shutil.rmtree(temp_raw_img_folder)

        # remove the records .csv too as we don't need it anymore - this avoids redownloading as well
        os.remove(records_path)

def splitCSVToFolder(csv_file_path, chunk_size, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load the CSV in chunks and save each chunk separately
    chunk_iterator = pd.read_csv(csv_file_path, chunksize=chunk_size)

    for i, chunk in enumerate(chunk_iterator):
        # Define the path for the chunk CSV file
        chunk_file_path = os.path.join(output_folder, f"chunk_{i + 1}.csv")

        # Save the chunk as a separate CSV file
        chunk.to_csv(chunk_file_path, index=False)

downloadImagesUsingDWC(data_folder="D:/bcc/all_beetles_download")