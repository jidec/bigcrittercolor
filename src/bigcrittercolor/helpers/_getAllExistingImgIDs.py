import os

def _getAllExistingImgIDs(data_folder):
    all_image_names = os.listdir(data_folder + "/all_images")
    img_ids = [name.replace(".jpg", "") for name in all_image_names]
    return(img_ids)
