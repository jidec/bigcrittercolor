import random
from bigcrittercolor import inferMasks
from bigcrittercolor.helpers import _getIDsInFolder, _loadTrainClassModel
from bigcrittercolor import clusterExtractSegs
import os
import shutil

def createInitialFilter(test_proportion=0.2, equalize_good_bad=True, data_folder=''):

    # get all existing masks
    mask_ids = _getIDsInFolder(data_folder + "/masks")

    # get good ids (all existing segments)
    good_ids = _getIDsInFolder(data_folder + "/segments")

    # get bad ids (all masks that weren't chosen for segments)
    bad_ids = list(set(mask_ids) - set(good_ids))

    # equalize the number of good and bad ids to balance the set
    if equalize_good_bad:
        bad_ids = random.sample(bad_ids,len(good_ids))

    def copy_proportion_traintest_ids_to_trainfolder(ids,testprop,goodbadstr):
        # get index dividing at proportion
        split_index = int(len(ids) * testprop)
        test_ids = ids[:split_index]
        train_ids = ids[split_index:]

        for id in test_ids:
            loc = data_folder + "/all_images/" + id + ".jpg"
            dest = data_folder + "/other/filter_training/test/" + goodbadstr + "/" + id + ".jpg"
            shutil.copy(loc, dest)
        for id in train_ids:
            loc = data_folder + "/all_images/" + id + ".jpg"
            dest = data_folder + "/other/filter_training/test/" + goodbadstr + "/" + id + ".jpg"
            shutil.copy(loc, dest)

    # copy images to folders
    copy_proportion_traintest_ids_to_trainfolder(good_ids,test_proportion,goodbadstr="good")
    copy_proportion_traintest_ids_to_trainfolder(bad_ids, test_proportion,goodbadstr="bad")

    _loadTrainClassModel(training_folder=data_folder + "/other/filter_training",
                        num_epochs=15, batch_size=6, num_workers=0,
                        data_transforms=None,
                        model_name="goodbadfilter")

    # test proportion