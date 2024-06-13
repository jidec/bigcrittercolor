import numpy as np
import cv2
import random

from bigcrittercolor.helpers.verticalize import _verticalizeImg
from bigcrittercolor.helpers import _showImages, _bprint, _readBCCImgs, _clusterByImgFeatures, _getBCCIDs, _writeBCCImgs
from bigcrittercolor.project import showBCCImages
from bigcrittercolor.helpers.image import _blobPassesFilter, _maskIsEmpty, _format

# Image - the raw starting images, not changed in any way
# Mask - masks that apply to the original image to yield a segment, created by inferMasks

# Segment - a "rotation invariant" object masked out of an image
#   Binary Segment - a blob that is simply white against black
#   Grey Segment - a blob that is greyscale
#   RGB Segment - a blob that is rgb

# Pattern - segments reduced to color data

# Extract segments using masks, normalize them, filter them using simple metrics, cluster using features from a pretrained CNN feature extractor
#@profile
def filterExtractSegs(img_ids=None, sample_n=None, batch_size=None,
    color_format_to_cluster = "grey", used_aux_segmodel=False,
    filter_hw_ratio_minmax = None, filter_prop_img_minmax = None, filter_symmetry_min = None, filter_intersects_sides=True, # filters
    mask_normalize_params_dict={'lines_strategy':"ellipse"}, # normalization/verticalization of masks
    feature_extractor="resnet18", # feature extractor
    cluster_params_dict={'algo':"kmeans",'n':4,'eps':0.1,'min_samples':24}, preselected_clusters_input = None,
    show=True, show_indv=False, print_steps=True, data_folder=""):

    """ Extract segments using masks, filter them, cluster them, keep clusters based on user input, then save kept segments

        Args:
            img_ids (list): the image IDs to use
            filter_hw_ratio_minmax (tuple): tuple of the min and max height-to-width ratios of verticalized masks to keep - masks below the min or above the mask are not used
            filter_prop_img_minmax (tuple): tuple of the min and max proportion of image taken up by the mask - masks taking up less than the min or greater than the max are not used
            filter_symmetry_min (float): float of the min symmetry score
            normalize_strategy (str): how we normalize masks for clustering and segment extraction.
                "vert_by_polygon" rotates the masks to be facing upward using an axis found by fitting a polygon to the mask blob
                    and finding the longest line through it between polygon points. we call this rotation normalization "verticalization"
                "vert_by_ellipse" verticalize masks using the major axis of an ellipse fit to the mask
                "vert_by_symmetry" verticalize masks using the line of best symmetry crossing through the mask blob centroid
                "vert_by_sym_if_sym" verticalize by symmetry ONLY if the masks are symmetrical above a certain threshold,
                    otherwise vert by polygon
            normalize_crop_sides_size (bool): a unique preprocessing step intended to trim wings and appendages off by cropping the sides of a blob on its axis
            preselected_clusters_input (str): the cluster selection input to feed when a user will not be present of the form "1,2,3" etc.
            feature_extractor (str): the name of the implemented feature extractor CNN - can be "inceptionv3", "vgg16", or "resnet18"
            cluster_algo (str): the name of the implemented clustering algorithm - can be "kmeans","agglom","gaussian_mixture", or "dbscan"
            cluster_n (int): number of clusters - if None a number is found using cluster dispersion criterion but this can take awhile
    """

    # if no ids specified load existing masks
    # note that if an aux segmodel was used, the images in masks are actually pre-extracted segments
    if img_ids is None:
        img_ids = _getBCCIDs(type="mask",data_folder=data_folder,sample_n=sample_n)

    # batching is necessary when cluster filtering large numbers of images (>10000)
    # not batching in this case will overload memory in the clustering step
    batches = [img_ids]
    if batch_size is not None:
        # split list into list of lists (batches)
        batches = []
        # iterate over the image IDs, stepping by the batch size
        for i in range(0, len(img_ids), batch_size):
            batches.append(img_ids[i:i + batch_size])

    for img_ids in batches:
        _bprint(print_steps,"Filtering " + str(len(img_ids)) + " segments by specified criterion...")
        masks = []
        mask_ids = []
        failed_hw_ids = []
        failed_prop_ids = []
        failed_sym_ids = []
        failed_edge_ids = []

        # for each image id
        for index, id in enumerate(img_ids):
            if index % 100 == 0:
                _bprint(print_steps, str(index) + "/" + str(len(img_ids)))

            # read image
            mask = _readBCCImgs(id, type="mask",data_folder=data_folder)

            # may have to convert this to binary

            # if mask fails to load at all
            if mask is None: continue

            # if image has no white (mask is empty)
            if _maskIsEmpty(mask):
                continue

            # keep in mind that currently these filters include verticalize I believe
            # prop filter
            if filter_prop_img_minmax is not None:
                if not _blobPassesFilter(mask, prop_img_minmax=filter_prop_img_minmax, prevert=used_aux_segmodel, show=show_indv):
                    failed_prop_ids.append(id)
                    continue

            # vert hw ratio filter
            if filter_hw_ratio_minmax is not None:
                if not _blobPassesFilter(mask, hw_ratio_minmax=filter_hw_ratio_minmax, prevert=used_aux_segmodel, show=show_indv):
                    failed_hw_ids.append(id)
                    continue

            # sym filter
            if filter_symmetry_min is not None:
                if not _blobPassesFilter(mask, rot_invar_sym_min=filter_symmetry_min, prevert=used_aux_segmodel, show=show_indv):
                    failed_sym_ids.append(id)
                    continue

            if filter_intersects_sides:
                if not _blobPassesFilter(mask, intersects_sides=True):
                    failed_edge_ids.append(id)
                    continue

            masks.append(mask)
            mask_ids.append(id)

        # print and show info about which masks passed
        _bprint(print_steps, str(len(masks)) + " masks passed all filters")
        show_type = "mask"
        if filter_prop_img_minmax is not None:
            _bprint(print_steps, str(len(failed_prop_ids)) + " masks failed for proportion of image")
            if show: showBCCImages(img_ids=failed_prop_ids, show_type=show_type, sample_n=18, title="Failed Prop",data_folder=data_folder)
        if filter_hw_ratio_minmax is not None:
            _bprint(print_steps, str(len(failed_hw_ids)) + " masks failed for polygon-verticalized height-to-width ratio")
            if show: showBCCImages(img_ids=failed_hw_ids, show_type=show_type, sample_n=18, title="Failed H/W",data_folder=data_folder)
        if filter_symmetry_min is not None:
            _bprint(print_steps, str(len(failed_sym_ids)) + " masks failed for symmetry")
            if show: showBCCImages(img_ids=failed_sym_ids, show_type=show_type, sample_n=18, title="Failed Sym",data_folder=data_folder)
        if filter_intersects_sides:
            _bprint(print_steps, str(len(failed_edge_ids)) + " masks failed for edge intersection")
            if len(failed_edge_ids) > 0:
                if show: showBCCImages(img_ids=failed_edge_ids, show_type=show_type, sample_n=18, title="Failed Edge",data_folder=data_folder)

        masks_unchanged = masks.copy() # save unchanged masks to use for seg extraction later

        # if aux segmodel was used, we would take the segments in masks folder and greyscale or binarize them
        # otherwise we would get the raw segs then either greyscale or binarize them
        # note that make_3channel is True because verticalizeImg takes 3 channel images
        if used_aux_segmodel:
            segs = _readBCCImgs(img_ids=mask_ids,type="mask",color_format=color_format_to_cluster,make_3channel=True, data_folder=data_folder)
        else:
            segs = _readBCCImgs(img_ids=mask_ids,type="raw_seg",color_format=color_format_to_cluster,make_3channel=True,data_folder=data_folder)

        _bprint(print_steps,"Normalizing segments for feature and segment extraction if specified...")
        if not used_aux_segmodel:
            for index, seg in enumerate(segs):
                if index % 100 == 0:
                    _bprint(print_steps, str(index) + "/" + str(len(img_ids)))
                segs[index] = _verticalizeImg(seg, **mask_normalize_params_dict)

        _bprint(print_steps, "Clustering masks by CNN-extracted features...")

        ids_for_full_display = mask_ids.copy()
        # if we used an aux segmodel, full display breaks so we give it None
        if used_aux_segmodel:
            ids_for_full_display = None

        # show is always true here because we have to show for user input
        labels = _clusterByImgFeatures(segs, feature_extractor=feature_extractor,
                                       full_display_ids=ids_for_full_display, full_display_data_folder=data_folder,
                                       print_steps=print_steps, cluster_params_dict=cluster_params_dict, show=True)

        if preselected_clusters_input is None:
            # take user input selecting the cluster number to extract segments for
            chosen_cluster_labels = input("Enter one or more cluster labels separated by commas and segments will be extracted for all images of those labels: ")
            chosen_cluster_labels = chosen_cluster_labels.split(',')
        else:
            # if we give it a string of preselected clusters use those of course
            chosen_cluster_labels = preselected_clusters_input.split(',')

        indices = []
        # get all indices matching the chosen cluster labels
        for index, label in enumerate(labels):
            if any(str(chosen_label) == str(label) for chosen_label in chosen_cluster_labels):
                indices.append(index)

        kept_ids = [mask_ids[i] for i in indices]

        # if we used an aux segmodel, we can just write the kept RGB segs in "masks" to segments
        if used_aux_segmodel:
            segs = _readBCCImgs(kept_ids, type="mask", data_folder=data_folder)

        # if we did not use an aux segmodel, we get the masks for the ids, apply them to the original images, and verticalize them
        else:
            masks = _readBCCImgs(kept_ids, type="mask", data_folder=data_folder)

            # temporary fix to probably a jpg corruption issue
            #for index, mask in enumerate(masks):
            #    mask[(mask > 10)] = 255
            #    mask[(mask < 10)] = 0
            #    masks[index] = mask

            # get all images to match the masks
            parent_imgs = _readBCCImgs(kept_ids, type="img", data_folder=data_folder)

            # zip masks and their parents
            masks_parents = [(x, y) for x, y in zip(masks, parent_imgs)]

            # extract segments using masks
            segs = [cv2.bitwise_and(parent_img, parent_img, mask=cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY).astype(np.uint8)) for mask, parent_img in masks_parents]

            #seg = np.copy(segs[0])
            #seg[np.where((seg == [0, 0, 0]).all(axis=2))] = [0, 0, 255]
            #cv2.imshow("0",seg)
            #cv2.waitKey(0)

            segs = [_verticalizeImg(seg, **mask_normalize_params_dict) for seg in segs]

            #seg = np.copy(segs[0])
            #seg[np.where((seg == [0, 0, 0]).all(axis=2))] = [0, 0, 255]
            #cv2.imshow("1", seg)
            #cv2.waitKey(0)

            # make seg 4 channel
            #segs = [_format(seg,)]

        # build imgnames
        kept_imgnames = [img_id + "_segment.png" for img_id in kept_ids]
        # write
        _writeBCCImgs(imgs=segs,imgnames=kept_imgnames,data_folder=data_folder)

        _bprint(print_steps, "Finished (batch)")
    _bprint(print_steps, "Finished (all batches)")
def _claheEqualize(img):
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

#if __name__ == '__main__':
#    filterExtractSegs(img_ids=_getIDsInFolder("E:/aeshna_data/masks", sample_n=5000), used_aux_segmodel=True,
#                       filter_hw_ratio_minmax=(3, 100), cluster_params_dict={'pca_n': None},
#                       filter_prop_img_minmax=(0.01, 0.9),
#                       data_folder="E:/aeshna_data")