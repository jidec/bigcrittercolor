import subprocess
import os
import shutil

from bigcrittercolor.project import setupBioencoderTrainingFolder

# wrapper around BioEncoder command line
def trainBioEncoder(min_imgs_per_class=20, max_imgs_per_class=100, run_name="v1", validation_percent=0.2, print_steps=True, data_folder=''):

    """ Train a BioEncoder model for metric learning of species-contrastive features
        This is a wrapper around package BioEncoder (Luerig et. al 2024) for easy application to bigcrittercolor data

            Args:
                min_imgs_per_class (str): minimum number of images required for a class for it to be used in training
    """

    setupBioencoderTrainingFolder(data_folder=data_folder, min_imgs_per_class=min_imgs_per_class, max_imgs_per_class = max_imgs_per_class, print_steps = print_steps)

    # bioencoder_configure
    root_dir = data_folder + "/other/bioencoder_training/bioencoder_wd"

    # bioencoder_split_dataset
    image_dir = data_folder + "/other/bioencoder_training/data_raw/aligned_train_val"

    # bioencoder_train
    val_percent = validation_percent

    config_path = data_folder + "/other/bioencoder_training/bioencoder_configs/train_stage1.yml"

    train_data_path = data_folder + "/other/bioencoder_training/bioencoder_wd/data/v1/train"
    val_data_path = data_folder + "/other/bioencoder_training/bioencoder_wd/data/v1/val"
    if os.path.exists(train_data_path):
        shutil.rmtree(train_data_path)
    if os.path.exists(val_data_path):
        shutil.rmtree(val_data_path)

    # define all commands
    commands = [
        [
            "bioencoder_configure",
            "--root-dir", root_dir,
            "--run-name", run_name
        ],
        [
            "bioencoder_split_dataset",
            "--image-dir", image_dir,
            "--val_percent", str(val_percent)
        ],
        [
            "bioencoder_train",
            "--config-path", config_path,
            "--overwrite"
        ]
    ]

    # Run each command sequentially
    for command in commands:
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error during execution of {' '.join(command)}: {e.stderr}")
            break  # Stop if any command fails

#trainBioencoder(min_imgs_per_class=5,max_imgs_per_class=100,data_folder="D:/bcc/ringtails")

# bugs
# timm_resnet18 works, prev doesnt
# terminal command breaks
# out of memory