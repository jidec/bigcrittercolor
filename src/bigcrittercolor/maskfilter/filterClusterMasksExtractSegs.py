from bigcrittercolor.helpers import _extractVerticalizeMasksOrSegs, _clusterByImgFeatures
import torch
from torchvision import models, transforms
import numpy as np
import os
import cv2
from sklearn.cluster import AffinityPropagation
from torch import optim, nn
from bigcrittercolor.helpers import _showImages, _bprint, _getIDsInFolder

# filter masks using simple metrics, then cluster using features from a pretrained CNN feature extractor
def filterClusterMasksExtractSegs(img_ids, filter_hw_ratio_minmax = (3, 100), filter_prop_img_minmax =(0, 0.15),
                        trim_sides_on_axis=False, clahe_equalize_segments = True, preselected_clusters_input =None,
                        affprop_pref=-50,affprop_damping = 0.7,
                       show=True, show_indv=False, print_steps=True, print_details=False, data_folder=""):

    # if no ids specified load existing masks
    if img_ids is None:
        img_ids = _getIDsInFolder(data_folder + "/masks")

    # turn list of ids into list of file locations
    img_locs = img_ids.copy()
    for i in range(0,len(img_locs)):
        img_locs[i] = data_folder + "/masks/" + img_locs[i] + "_mask.jpg"

    _bprint(print_steps,"Filtering masks by height/width ratio and proportion of image covered...")
    masks = []
    # save mask ids too
    mask_ids = []
    # for each image location
    for index, img_loc in enumerate(img_locs):

        _bprint(print_details, "Loading " + img_loc + "...")
        mask = cv2.imread(img_loc, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            _bprint(print_details, "File not found, skipping...")
            continue

        if show_indv:
            cv2.imshow("Mask",mask)
            cv2.waitKey(0)

        if filter_prop_img_minmax is not None:

            white_pixel_count = cv2.countNonZero(mask)

            total_area = mask.shape[0] * mask.shape[1]

            white_area_percent = (white_pixel_count / total_area)

            if white_area_percent < filter_prop_img_minmax[0]:
                if print_details: print("Failed: mask percent image " + str(white_area_percent) + " area less than min")
                continue
            if white_area_percent > filter_prop_img_minmax[1]:
                if print_details: print("Failed: mask percent image " + str(white_area_percent) + " area greater than max")
                continue

        if filter_hw_ratio_minmax is not None:
            # rotate to vertical (only works on rgb masks)
            #rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            vert = _extractVerticalizeMasksOrSegs([mask])

            # if failed, skip
            if vert is None:
                continue
            else:
                # if hw_ratio less than min or greater than max, skip
                vert = vert[0]
                h = np.shape(vert)[0]
                w = np.shape(vert)[1]
                hw_ratio = h / w
                if hw_ratio < filter_hw_ratio_minmax[0]:
                    if print_details: print("Failed: hw ratio " + str(hw_ratio) + " less than min")
                    continue
                if hw_ratio > filter_hw_ratio_minmax[1]:
                    if print_details: print("Failed: hw ratio " + str(hw_ratio) + "  greater than max")
                    continue

        # TODO can show a sample of masks that passed the filter
        masks.append(mask)
        mask_ids.append(img_ids[index])

    masks_unchanged = masks.copy()

    _bprint(print_steps,"Verticalizing masks to normalize them for feature extraction...")
    masks = _extractVerticalizeMasksOrSegs(masks,show=show_indv)

    # cluster by features and return cluster labels
    # show is always true here because we have to show for user input
    labels = _clusterByImgFeatures(masks, print_steps=print_steps, show=True,affprop_damping=affprop_damping, affprop_pref=affprop_pref)

    if preselected_clusters_input is None:
        # take user input selecting the cluster number to extract segments for
        chosen_cluster_labels = input("Enter one or more cluster labels separated by commas and segments will be extracted for all images of those labels: ")
        chosen_cluster_labels = chosen_cluster_labels.split(',')
    else:
        chosen_cluster_labels = preselected_clusters_input.split(',')

    indices = []
    # get all indices matching the chosen cluster label
    for index, label in enumerate(labels):
        if any(str(chosen_label) == str(label) for chosen_label in chosen_cluster_labels):
        #if str(label) == str(chosen_cluster_label):
            indices.append(index)

    # get all masks matching the index (using the non-verticalized masks we saved)
    masks = [masks_unchanged[i] for i in indices]

    # temporary fix to probably a jpg corruption issue
    for index, mask in enumerate(masks):
        mask[(mask > 10)] = 255
        mask[(mask < 10)] = 0
        masks[index] = mask

    # get all ids matching the index
    img_ids = [mask_ids[i] for i in indices]

    # get all images to match the masks
    parent_imgs = [cv2.imread(data_folder + "/all_images/" + img_id + ".jpg") for img_id in img_ids]

    # zip masks and their parents
    masks_parents = [(x, y) for x, y in zip(masks, parent_imgs)]
    for tuple_images in masks_parents:
        for image in tuple_images:
            if show_indv:
                cv2.imshow("0",image)
                cv2.waitKey(0)

    # extract segments using masks
    segments = [cv2.bitwise_and(parent_img, parent_img, mask=mask.astype(np.uint8)) for mask, parent_img in masks_parents]

    # extract and verticalize them
    segments = _extractVerticalizeMasksOrSegs(segments)
    #segments = [claheEqualize(seg) for seg in segments]

    # zip segments and their ids
    segs_ids = [(x, y) for x, y in zip(segments, img_ids)]

    # write each segment naming it by its ID
    for seg_and_id in segs_ids:
        dest = data_folder + "/segments/" + seg_and_id[1] + "_segment.png"
        if show_indv:
            cv2.imshow("1",seg_and_id[0])
            cv2.waitKey(0)
        cv2.imwrite(dest, seg_and_id[0])

    _bprint(print_steps, "Finished")

class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        # Extract VGG-16 Feature Layers
        self.features = list(model.features)
        self.features = nn.Sequential(*self.features)
        # Extract VGG-16 Average Pooling Layer
        self.pooling = model.avgpool
        # Convert the image into one-dimensional vector
        self.flatten = nn.Flatten()
        # Extract the first part of fully-connected layer from VGG16
        self.fc = model.classifier[0]

    def forward(self, x):
        # It will take the input 'x' until it returns the feature vector called 'out'
        out = self.features(x)
        out = self.pooling(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out

def claheEqualize(img):
    # We first create a CLAHE model based on OpenCV
    # clipLimit defines threshold to limit the contrast in case of noise in our image
    # tileGridSize defines the area size in which local equalization will be performed
    clahe_model = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    # For ease of understanding, we explicitly equalize each channel individually
    colorimage_b = clahe_model.apply(img[:, :, 0])
    colorimage_g = clahe_model.apply(img[:, :, 1])
    colorimage_r = clahe_model.apply(img[:, :, 2])

    # Next we stack our equalized channels back into a single image
    colorimage_clahe = np.stack((colorimage_b, colorimage_g, colorimage_r), axis=2)

    # Using Numpy to calculate the histogram
    #color = ('b', 'g', 'r')
    #for i, col in enumerate(color):
    #    histr, _ = np.histogram(colorimage_clahe[:, :, i], 256, [0, 256])
    #    plt.plot(histr, color=col)
    #    plt.xlim([0, 256])
    #plt.show()

    return(colorimage_clahe)