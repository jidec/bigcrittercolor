import pandas as pd
import os
import cv2
import shutil
import yaml

from bigcrittercolor.helpers import _readBCCImgs, _getBCCIDs
from bigcrittercolor.helpers.ids import _imgIDToObsID

def setupBioencoderTrainingFolder(img_ids=None, data_folder='', min_imgs_per_class=20, max_imgs_per_class=100, img_size=None, batch_size=None, n_workers=None):

    if img_ids is None:
        img_ids = _getBCCIDs(type="segment",data_folder=data_folder)

    # read images and records
    imgs = _readBCCImgs(img_ids=img_ids, type="segment", data_folder=data_folder)
    records = pd.read_csv(data_folder + '/records.csv')

    # create new folder "bioencoder_training" in "data_folder/other" - it will contain many subfolders
    new_folder = os.path.join(data_folder, "other", "bioencoder_training")
    # define the subfolder names and their nested folders
    subfolders = {
        "bioencoder_configs": [],
        "bioencoder_wd": ["data", "logs", "plots", "runs", "weights"],
        "data_raw": ["aligned_train_val"]
    }

    # delete the previous training folder if we already made one
    if os.path.exists(new_folder):
        shutil.rmtree(new_folder)

    os.makedirs(new_folder)
    # create subfolders
    for subfolder, nested_folders in subfolders.items():
        subfolder_path = os.path.join(new_folder, subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
        # Create nested folders within each subfolder
        for nested in nested_folders:
            nested_path = os.path.join(subfolder_path, nested)
            if not os.path.exists(nested_path):
                os.makedirs(nested_path)

    obs_ids = [_imgIDToObsID(img_id) for img_id in img_ids]
    # move images of each species to "bioencoder_training/data_raw/aligned_train_val/*species*"
    species_image_count = {}
    for img, img_id, obs_id in zip(imgs, img_ids, obs_ids):
        try:
            # Find the species for the given image ID
            species = records[records['obs_id'] == obs_id]['species'].values[0]
        except IndexError:
            continue
        species = species.replace(" ", "_")
        # Create a directory for the species if it doesn't exist
        species_dir = os.path.join(new_folder, "data_raw", "aligned_train_val", species)
        if not os.path.exists(species_dir):
            os.makedirs(species_dir)
        # Initialize image count for the species if necessary
        if species not in species_image_count:
            species_image_count[species] = 0
        # Save the image only if the count is less than max_imgs_per_class
        if species_image_count[species] < max_imgs_per_class:
            image_path = os.path.join(species_dir, f"{img_id}.png")
            cv2.imwrite(image_path, img)
            species_image_count[species] += 1
    # Clean up directories with fewer than min_imgs_per_class
    for species, count in species_image_count.items():
        species_dir = os.path.join(new_folder, "data_raw", "aligned_train_val", species)
        if count < min_imgs_per_class:
            shutil.rmtree(species_dir)

    # write default bioencoder configs
    # define the content of the .yml file
    config_content = {
        "model": {
            "backbone": "timm_tf_efficientnet_b5.ns_jft_in1k",
            "ckpt_pretrained": None,
            "num_classes": "in the first stage of training we don't need num_classes, since we don't have a FC head"
        },
        "train": {
            "n_epochs": 100,
            "amp": True,
            "ema": True,
            "ema_decay_per_epoch": 0.4,
            "target_metric": "precision_at_1",
            "stage": "first"
        },
        "dataloaders": {
            "train_batch_size": 20,
            "valid_batch_size": 20,
            "num_workers": 5
        },
        "optimizer": {
            "name": "SGD",
            "params": {
                "lr": 0.003
            }
        },
        "scheduler": {
            "name": "CosineAnnealingLR",
            "params": {
                "T_max": 100,
                "eta_min": 0.0003
            }
        },
        "criterion": {
            "name": "SupCon",
            "params": {
                "temperature": 0.1
            }
        },
        "img_size": 200,
        "augmentations": {
            "transforms": [
                {"RandomResizedCrop": {"height": 200, "width": 200, "scale": (0.7, 1)}},
                "Flip",
                "RandomRotate90",
                {"MedianBlur": {"blur_limit": 3, "p": 0.3}},
                {"ShiftScaleRotate": {"p": 0.4}},
                "OpticalDistortion",
                "GridDistortion",
                "HueSaturationValue"
            ]
        }
    }

    # write the content to the .yml file in the bioencoder_configs folder
    config_folder = os.path.join(new_folder, "bioencoder_configs")
    config_file_path = os.path.join(config_folder, "train_stage1.yml")

    with open(config_file_path, 'w') as file:
        yaml.dump(config_content, file, default_flow_style=False)

#setupBioencoderTrainingFolder(data_folder="D:/bcc/ringtails")
