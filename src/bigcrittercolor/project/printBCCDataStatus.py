from bigcrittercolor.helpers.ids import _getIDsInFolder
from bigcrittercolor.helpers import _getBCCIDs
import glob
import pandas as pd
import os

def printBCCDataStatus(data_folder):

    """ Print information about a bigcrittercolor data folder, such as the number of raw images, masks, and segments

        Args:
            data_folder (str): location of the bigcrittercolor data folder
    """

    def get_str_percent_rounded(num, denom):
        return(str(round((num/denom) * 100,0)))

    print("Status of bigcrittercolor data folder at " + data_folder + "...")
    print()

    # download records
    path = data_folder + "/other/inat_download_records/"
    pattern = f"{path}/iNat*.csv"
    total_rows = 0
    # Find all matching CSV files
    csv_files = glob.glob(pattern)
    # Iterate over the list of file paths & read each file
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        total_rows += len(df)
    n_recs = total_rows
    n_obs = df['obs_id'].nunique()
    print("Number of downloaded observation records: " + str(n_obs))

    # List to hold the filenames
    filenames_starting_with_iNat = []
    path = data_folder + "/other/inat_download_records"
    # Iterate over the entries in the directory
    for entry in os.listdir(path):
        # Check if the entry is a file and if its name starts with "iNat"
        if os.path.isfile(os.path.join(path, entry)) and entry.startswith("iNat"):
            filenames_starting_with_iNat.append(entry)
    taxa_names = [s.split('-')[1].split('.')[0] for s in filenames_starting_with_iNat]
    print("Taxa with downloaded records: " + str(taxa_names))
    print()

    n_imgs = len(_getBCCIDs(type="image",data_folder=data_folder))
    print("Number of downloaded images: " + str(n_imgs))
    print("Percent of records with images: " + get_str_percent_rounded(n_imgs,n_recs))
    print()

    n_masks = len(_getBCCIDs(type="mask",data_folder=data_folder))
    print("Number of inferred masks: " + str(n_masks))
    print("Percent of images with masks: " + get_str_percent_rounded(n_masks,n_imgs))
    print()

    n_segs = len(_getBCCIDs(type="segment",data_folder=data_folder))
    print("Number of segments extracted after filtering: " + str(n_segs))
    print("Percent of images with segments: " + get_str_percent_rounded(n_segs, n_imgs))
    print()

    n_pats = len(_getIDsInFolder(data_folder + "/patterns"))
    print("Number of final patterns: " + str(n_pats))
    print("Percent of images with patterns: " + get_str_percent_rounded(n_pats, n_imgs))

    # has goodbadclassifier been created?
    # number of segments attempted vs number of good masks