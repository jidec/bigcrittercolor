from bigcrittercolor.helpers import _getIDsInFolder

def printBCCDataStatus(data_folder):
    print("Status of bigcrittercolor data folder at " + data_folder)

    n_imgs = len(_getIDsInFolder(data_folder + "/all_images"))
    print("Num downloaded images: " + str(n_imgs))

    n_masks = len(_getIDsInFolder(data_folder + "/masks"))
    print("Num inferred masks: " + str(n_masks))

    n_segs = len(_getIDsInFolder(data_folder + "/segments"))
    print("Num segments extracted: " + str(n_segs))

    # has goodbadclassifier been created?
    # number of segments attempted vs number of good masks
