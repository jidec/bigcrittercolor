import os
import pandas as pd
import cv2
import shutil
import zipfile
import csv

from bigcrittercolor.helpers import _bprint, _inat_dl_helpers, _rebuildiNatRecords, _writeBCCImgs, _getBCCIDs, _mkDirsIfNotExists
import xml.etree.ElementTree as ET

def downloadImagesUsingDarwinCore(dwc_archive_location,
                          inat_img_size="medium",
                          download_n_per_species=None,
                          download_chunk_size=5000,
                          skip_existing=True,
                          print_steps=True, data_folder=''):

    """ Download images using a DarwinCore archive you've downloaded from GBIF - recommended when downloading many taxa.

        Works with iNaturalist and Observation.org for now, with other GBIF data sources to be added later.

        Images are saved in data_folder/all_images unless LMDB option was selected during project folder creation.

        Args:
            dwc_archive_location (str): location of the DarwinCore archive
            inat_img_size (str): size of the iNaturalist images, can be "medium" or "original"
            download_n_per_species (int): max number of images to download for a species - i.e. if a species has more than 200, only keep 200
    """
    # setup paths
    records_csv_path = os.path.join(data_folder, "records.csv")
    darwincore_folder_path = os.path.join(data_folder, "other", "gbif_darwincore_records_split")

    # only create records if either of the paths does not exist - so we only do this once
    if not os.path.exists(records_csv_path) or not os.path.exists(darwincore_folder_path):
        #_bprint(print_steps, "Creating records for image download using unzipped DarwinCore archive in other/my_gbif_darwincore...")
        _bprint(print_steps, "Creating records for image download using zipped DarwinCore archive at " + dwc_archive_location + "...")

        # get name of DarwinCore folder as the only folder in other/my_gbif_darwincore
        #other_folder = os.path.join(data_folder, "other", "my_gbif_darwincore")
        #folder_names = [name for name in os.listdir(other_folder) if os.path.isdir(os.path.join(other_folder, name))]
        #if len(folder_names) == 0:
        #    raise FileNotFoundError("No DarwinCore archive folder found in other/my_gbif_darwincore, did you add it?")

        _bprint(print_steps, "Reading occurrences and multimedia links... ")
        # open the zipped DWC archive
        with zipfile.ZipFile(dwc_archive_location, 'r') as zip_ref:
            # read multimedia (image links)
            with zip_ref.open("multimedia.txt") as file:
                multimedia_df = pd.read_csv(file, sep='\t')
            # read occurrences (data)
            with zip_ref.open("occurrence.txt") as file:
                # read in chunks to avoid errors associated with reading large csvs
                chunk_size = 100000  # Try smaller or larger values if needed
                chunks = []
                for chunk in pd.read_csv(file, sep='\t', quoting=csv.QUOTE_NONE, low_memory=True,
                                         chunksize=chunk_size):
                    chunks.append(chunk)
                occurrence_df = pd.concat(chunks, ignore_index=True)
            # read metadata to save DOI used for download
            with zip_ref.open("metadata.xml") as file:
                # parse the metadata XML file
                tree = ET.parse(file)
                root = tree.getroot()
                # extract the 'packageId' attribute, which is the DOI
                package_id = root.attrib.get('packageId')
                # write the packageId to a file
                with open(data_folder + '/other/processing_info/used_darwincore_doi.txt', 'w') as f:
                    f.write(package_id + '\n')

        # define the paths to the TSV files
        #multimedia_path = os.path.join(target_folder, "multimedia.txt")
        #occurrence_path = os.path.join(target_folder, "occurrence.txt")

        # load the TSV files into pandas DataFrames
        #multimedia_df = pd.read_csv(multimedia_path, sep='\t')
        #occurrence_df = pd.read_csv(occurrence_path, sep='\t')

        _bprint(print_steps, "Dropping duplicates and merging... ")
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
        "scientificName","kingdom","phylum","class","order","superfamily","family","genus","species","infraspecificEpithet","taxonomicStatus",
        "imageLink","rightsHolder"]]

        merged_df["imageLink"] = merged_df["imageLink"].str.replace("original", inat_img_size, regex=False)
        # create img_url and file_name cols required by downloader
        merged_df["img_url"] = merged_df["imageLink"]
        merged_df["file_name"] = merged_df["catalogNumber"]

        # keep only rows with image urls
        merged_df = merged_df.loc[merged_df['img_url'].notna() & (merged_df['img_url'] != "")]

        if download_n_per_species is not None:
            _bprint(print_steps, "Keeping " + str(download_n_per_species) + " images per species...")
            # keep at most 200 random images per species
            # species or scientificName?
            merged_df = merged_df.groupby('species').apply(lambda x: x.sample(min(len(x), download_n_per_species))).reset_index(
                drop=True)

        splitDataFrameToFolder(merged_df, download_chunk_size,output_folder=data_folder + "/other/split_gbif_download_records")
        # split records into .csvs which will be passed to the downloader (download chunks)
        # splitCSVToFolder(data_folder + "/records.csv", download_chunk_size, output_folder=data_folder + "/other/split_gbif_download_records")

        # now that records for download are set, make some final changes to records df
        # kind of a workaround so that the previous downloader still works
        # this includes some new cols and removed cols
        # some changes for Obs.org rows
        # add Observation.org to datasetName column
        merged_df.loc[merged_df['occurrenceID'].str.contains('observation.org', na=False), 'datasetName'] = 'Observation.org'
        # remove cols that downloader needed
        merged_df = merged_df.drop(columns=["img_url"])
        # add catalogNumber back in (gets dropped in Obs.org for some reason)
        merged_df['catalogNumber'] = merged_df['occurrenceID'].apply(lambda x: x.split('/')[-1])
        # add img_id column critical for bigcrittercolor
        merged_df['img_id'] = merged_df.apply(lambda x: f"OBS-{x['catalogNumber']}" if x['datasetName'] == 'Observation.org' else f"INAT-{x['catalogNumber']}",
                                    axis=1)

        # add file_name column
        merged_df['file_name'] = merged_df['img_id'] + '.jpg'

        # write to records in root
        merged_df.to_csv(data_folder + "/records.csv", index=False)

        _bprint(print_steps, "Finished processing DarwinCore archive - number of records with images to download: " + str(merged_df.shape[0]))

    # location of download log
    fileout = data_folder + "/other/download/download_log.csv"
    # location of split .csvs
    csv_folder = data_folder + "/other/split_gbif_download_records"
    # get a list of all CSV files in the directory
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

    _bprint(print_steps, "Started downloading chunks of images: " + str(len(csv_files)) + " remaining")
    # loop through each .csv file and download all the images in each
    for csv_file in csv_files:
        records_path = os.path.join(csv_folder, csv_file)
        print(f"Processing {csv_file}...")

        temp_raw_img_folder = data_folder + "/other/temp_raw_imgs"

        _mkDirsIfNotExists(temp_raw_img_folder)

        # download the images
        _inat_dl_helpers.downloadImages(img_records=records_path, imgdir=temp_raw_img_folder, fileout=fileout)

        # if records are iNaturalist...
        # img_name_prefix = "INAT"

        # rename raw images with prefix
        imgnames = os.listdir(temp_raw_img_folder)
        for name in imgnames:
            newname = name.split('_')[0]
            #os.rename(temp_raw_img_folder + "/" + name, temp_raw_img_folder + "/" + img_name_prefix + "-" + newname)
            os.rename(temp_raw_img_folder + "/" + name, temp_raw_img_folder + "/" + newname)

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

        # remove the records .csv too as we don't need it anymore - this means that completed chunks aren't redownloaded
        os.remove(records_path)

    _bprint(print_steps, "Finished downloadImagesUsingDarwinCore.")

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

def splitDataFrameToFolder(df, chunk_size, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Split the DataFrame into chunks and save each chunk separately
    for i, chunk in enumerate(range(0, len(df), chunk_size)):
        # Define the path for the chunk CSV file
        chunk_file_path = os.path.join(output_folder, f"chunk_{i + 1}.csv")

        # Get the chunk of the DataFrame
        df_chunk = df.iloc[chunk:chunk + chunk_size]

        # Save the chunk as a separate CSV file
        df_chunk.to_csv(chunk_file_path, index=False)

#downloadImagesUsingDWC(data_folder="D:/bcc/all_beetles_download_obsorg")