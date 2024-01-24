import pandas as pd
import os
import cv2
import shutil

from bigcrittercolor.helpers import _readBCCImgs, _getIDsInFolder
def makeBioencoderFolder(data_folder, min_imgs_per_class=50):
    # get ids of images i.e. INAT-31231-2
    img_ids = _getIDsInFolder(data_folder + "/segments")
    # get ids of observations i.e. INAT-32312
    obs_ids = [s.rsplit('-', 1)[0] for s in img_ids]

    imgs = _readBCCImgs(img_ids=img_ids, type="seg",data_folder=data_folder)
    records = pd.read_csv(data_folder + '/records.csv')

    new_folder = data_folder + "/bioencoder_data"
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    for img, img_id, obs_id in zip(imgs, img_ids, obs_ids):
        try:
            # Find the species for the given image id
            species = records[records['obs_id'] == obs_id]['species'].values[0]
        except IndexError:
            continue

        species = species.replace(" ", "_")

        # Create a directory for the species if it doesn't exist
        species_dir = os.path.join(new_folder, species)
        if not os.path.exists(species_dir):
            os.makedirs(species_dir)

        # Save the image to the directory
        image_path = os.path.join(species_dir, f"{img_id}.png")
        cv2.imwrite(image_path, img)

    # Iterate over all subdirectories in the given folder
    for subdir in os.listdir(new_folder):
        subdir_path = os.path.join(new_folder, subdir)

        # Ensure it's a directory and not a file
        if os.path.isdir(subdir_path):
            # Count the number of image files in the subdirectory
            image_count = len(os.listdir(subdir_path))

            # If image count is less than 50, remove the subdirectory
            if image_count < min_imgs_per_class:
                shutil.rmtree(subdir_path)