import cv2
import subprocess

from bigcrittercolor.helpers import _bprint, _getBCCIDs, _readBCCImgs, _getSequentialSavePath, _mergeAndSaveFolderCSVs, _mkDirsIfNotExists

def inferBioEncoderFeatures(img_ids=None, print_steps=True,data_folder=''):
    
    _bprint(print_steps, "Starting inferBioEncoderFeatures...")
    if img_ids is None:
        _bprint(print_steps, "No ids specified, getting existing segment ids...")
        img_ids = _getBCCIDs(type="segment", data_folder=data_folder)

    _bprint(print_steps, "Reading in images...")
    # read in images
    imgs = _readBCCImgs(img_ids,type="segment", data_folder=data_folder)

    _bprint(print_steps, "Writing images to folder for inference...")
    # write images
    for img, id in zip(imgs,img_ids):
        _mkDirsIfNotExists([data_folder + "/other/bioencoder", data_folder + "/other/bioencoder/images_to_infer"])
        cv2.imwrite(data_folder + "/other/bioencoder/images_to_infer/" + id + ".png",img)

    # add _1, _2 etc. to save path if necessary
    csv_save_path = _getSequentialSavePath(data_folder + "bioencodings.csv")

    _bprint(print_steps, "Running BioEncoder inference command...")
    # run the command
    commands = [
        [
            "bioencoder_inference",
            "--config-path", data_folder + "/other/bioencoder/bioencoder_training/bioencoder_configs/inference.yml",
            "--path", data_folder + "/other/bioencoder/images_to_infer",
            "--save-path", csv_save_path
        ]
    ]

    # run each command sequentially
    for command in commands:
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error during execution of {' '.join(command)}: {e.stderr}")
            break  # Stop if any command fails

    _bprint(print_steps, "Saving encodings to " + csv_save_path + "...")
    # combine csvs into one
    # could re-add this, but probably overcomplicated
    #_mergeAndSaveFolderCSVs(folder_path=data_folder + "/other/bioencoder/inferred_encodings",save_path=data_folder + "/bioencodings.csv")

    _bprint(print_steps, "Finished inferBioEncoderFeatures.")