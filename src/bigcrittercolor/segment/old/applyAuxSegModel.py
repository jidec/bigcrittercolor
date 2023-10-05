from bigcrittercolor.helpers import _getIDsInFolder

def applyAuxSegModel(img_ids=None, model_location='', data_folder=''):
    # if no ids get all existing
    if img_ids is None:
        img_ids = _getIDsInFolder(data_folder + "/all_images")
