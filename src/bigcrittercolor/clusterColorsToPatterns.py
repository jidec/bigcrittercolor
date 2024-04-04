import pandas as pd
import numpy as np
import cv2
from numpy import unique
import os
from collections import Counter
import random

from bigcrittercolor.helpers import _bprint, _getIDsInFolder, _showImages
from bigcrittercolor.helpers.clustering import _cluster
from bigcrittercolor.helpers.image import _blur, _format, _imgToColorPatches, _reconstructImgFromPPD, _blackBgToTransparent
from bigcrittercolor.helpers.image import _equalize
from bigcrittercolor.helpers import makeCollage

def clusterColorsToPatterns(img_ids=None, cluster_individually=False, preclustered = False, group_cluster_records_colname = None,
                    by_patches=True, patch_args = {'min_patch_pixel_area':5,'cluster_args':{'n':5, 'algo':'gaussian_mixture'}}, visualize_patching=True,
                    cluster_args={'find_n_minmax':(2,7), 'algo':"gaussian_mixture"}, use_positions=False,
                    colorspace = "cielab",
                    height_resize = 200,
                    equalize_args={'type':"clahe"},
                    blur_args= {'type':"bilateral"},
                    preclust_read_subfolder = "", write_subfolder= "",
                    batch_size = None,
                    print_steps=True, print_details=False,
                    show=True, show_indv=False, data_folder="../.."):

    """ Use clustering to reduce continuously shaded organismal segments to discrete patterns

        Args:
            img_ids (list): the imageIDs to use segments from
            preclustered (bool): specify whether you want to cluster preclustered data, applying the second step in a 2-part clustering run
            group_cluster_records_colname (str): the column in records to group images before clustering i.e "species"
                can also be "speciesMorph" which gets merged from inferences
            group_cluster_provided_ids (bool): whether to group cluster the provided ids as whole, rather than splitting them into groups with the group_cluster_records_colname
                OR just clustering each image individually
            by_patches (bool): whether to, instead of clustering segment pixels, reduce segments to color patches and cluster the mean colors of those patches
            min_patch_pixel_area (int): the minimum number of pixels in a patch to include it as a pattern element in clustering
            blur_args (dict): a dictionary of params including 'type' ('gaussian' or 'bilateral'), 'd', 'sigma_color', and 'sigma_space'
            cluster_args (dict): a dictionary of params including 'n' (the integer number of clusters) and
                'algo' (a string that can be 'kmeans', 'gaussian_mixture', or 'agglom')
                that get passed to the clustering helper function
            colorspace (str): the colorspace to cluster in, None uses rgb, "hls" uses HLS, and "lab" uses CIELAB
            use_positions (bool): whether to cluster using pixel positions within the image in addition to pixel values
                This means that pixels that are closer together will be more likely to be assigned to the same cluster

            downweight_axis: the index of the color axis (such as R,G,B or H,S,L) to downweight
            upweight_axis: the index of the color axis (such as R,G,B or H,S,L) to upweight
            height_resize: the number of pixels to resize to in height before pixels are extracted and clustered, keeping the image aspect ratio intact

            gaussian_blur: whether to apply gaussian blur to input images
        :param bool bilat_blur: whether to apply bilateral blur to input images - preserves edges very well
        :param int blur_size: param of bilateral blur
        :param int blur_sigma: param of bilateral blur

        :param bool print_steps: print processing step info after they are performed
        :param bool print_details: print very verbose details for each segment

        :param str preclust_read_subfolder: the folder to draw preclustered images from if using preclustered
        :param str write_subfolder: the folder to write discretized patterns to

        :param bool show: show image processing outputs and intermediates
        :param bool data_folder: the path to the project directory containing /data and /trainset_tasks folders
    """

    # get all ids if ids is None
    if img_ids is None:
        img_ids = _getIDsInFolder(data_folder + "/segments")
        if preclustered:
            img_ids = _getIDsInFolder(data_folder + "/patterns/" + preclust_read_subfolder)

    # this list holds either patch data or pixel data for every image - soon we will gather this
    patch_or_pixel_data = []

    # print expected amount of time the function should take - need to rework this
    #if preclustered:
    #    _bprint(print_steps, "Estimated time " + str(len(img_ids)/30) + " minutes at default cluster mode, metric, and vert resize")
    #else:
    #    _bprint(print_steps, "Estimated time " + str((len(img_ids)/30)/3) + " minutes at default cluster mode, metric, and vert resize")

    # load images by id
    _bprint(print_steps, "Gathering pixel or patch data for each image...")
    for i, id in enumerate(img_ids):
        if i % 100 == 0:
            print(i)
        if not preclustered: # if clustering normally, just read in segments
            img = cv2.imread(data_folder + "/segments/" + id + "_segment.png",cv2.IMREAD_UNCHANGED)
        else: # otherwise read in from the preclust_read_subfolder in patterns
            img = cv2.imread(data_folder + "/patterns/" + preclust_read_subfolder + "/" + id + "_pattern.png", cv2.IMREAD_UNCHANGED) #_pattern.ong

        _showImages(show_indv,[img],"Segment")

        _bprint(print_details, "Loaded segment for ID " + id)
        if img is None:
            _bprint(print_details, "Image is empty - skipping")
            continue

        # get mask using black pixels before blur
        # find the black background color, which will always be the most common color
        pixels = img.reshape(-1, img.shape[2])
        pixel_tuples = [tuple(pixel) for pixel in pixels]
        most_common_color, _ = Counter(pixel_tuples).most_common(1)[0]
        # create a mask for the black background
        bg_mask = np.all(img == most_common_color, axis=-1)
        # convert the boolean mask to uint8 format for visualization
        bg_mask = bg_mask.astype(np.uint8) * 255

        # blur
        if blur_args is not None:
            img = _blur(img, **blur_args,show=show_indv)

        # equalize
        if equalize_args is not None:
            img = _equalize(img, **equalize_args,show=show_indv)

        # format colorspace
        img = _format(img, in_format='rgb', out_format=colorspace,alpha=False)

        # resize
        if height_resize is not None:
            resize_proportion = height_resize / img.shape[0]
            img = cv2.resize(img, dsize=(int(img.shape[1] * resize_proportion),height_resize),interpolation = cv2.INTER_NEAREST)
            bg_mask = cv2.resize(bg_mask, dsize=(int(bg_mask.shape[1] * resize_proportion),height_resize),interpolation = cv2.INTER_NEAREST)

        # gather data for the image
        # patch_or_pixel_data is a list of tuples that allows us to reconstruct the image later
        #   each tuple contains: a list of locations of the pixels and patches as its first element
        #                        a list of the values of the pixels or patches as its second element
        #                        an ndarray of the image shape
        if by_patches: # patch data
            patch_or_pixel_data.append(_imgToColorPatches(img, bg_mask, **patch_args, return_patch_masks_colors_imgshapes=True,show=show_indv, input_colorspace=colorspace))
            _bprint(print_details, "Gathered image patch data")
        else: # pixel data
            patch_or_pixel_data.append(getPixelCoordsAndPixels(img))
            _bprint(print_details, "Gathered image pixel data")

    if by_patches and visualize_patching:
        patch_imgs = [_reconstructImgFromPPD(ppd,is_patch_data=True) for ppd in patch_or_pixel_data]
        patch_imgs = [_format(img,in_format=colorspace,out_format="rgb") for img in patch_imgs]
        if len(patch_imgs) > 18:
            indices = random.sample(range(len(patch_imgs)), 18)
            patch_imgs = [patch_imgs[i] for i in indices]
            raw_imgs = [cv2.imread(data_folder + "/segments/" + img_ids[i] + "_segment.png",cv2.IMREAD_UNCHANGED) for i in indices]
            patch_imgs = [makeCollage([img,patch_img],n_per_row=2,resize_wh=(50,200)) for img, patch_img in zip(raw_imgs,patch_imgs)]


        _showImages(True,patch_imgs,maintitle="Example Patched Images")

    # cluster by group
    if group_cluster_records_colname is not None:
        _bprint(print_steps, "Started group clustering by a records column (probably 'species' or 'genus')")
        # load records
        records = pd.read_csv(data_folder + "/records.csv")
        records = records[['obs_id', group_cluster_records_colname]]
        records = records.drop_duplicates(subset='obs_id', keep='first')

        # create df to hold img_ids and corresponding taxa
        img_ids_groups = pd.DataFrame(img_ids, columns=['img_id'])
        img_ids_groups['obs_id'] = ["-".join(id.split("-")[:-1]) for id in img_ids] # add obs_id column

        # merge in
        img_ids_groups = pd.merge(img_ids_groups, records, on='obs_id')

        # loop through unique groups and update patch data using clustered centroids
        unique_groups = unique(img_ids_groups[group_cluster_records_colname])
        updated_patch_or_pixel_data = []
        updated_img_ids = []
        for g in unique_groups:

            # get indices for the group
            group_indices = img_ids_groups.index[img_ids_groups[group_cluster_records_colname] == g].tolist()

            # get data for the group
            group_ppds = [patch_or_pixel_data[i] for i in group_indices]

            # get ids for the group
            group_img_ids = [img_ids[i] for i in group_indices]

            ## BELOW PASTED FROM GROUP CLUSTERING - may want to make own fun
            # get color values per image (a list of lists, each sublist containing RGB color values each of which has a length of 3)
            all_values_per_image = [ppd[1] for ppd in group_ppds]

            # combine the image sublists - all_values is a list of color values
            all_values = [value for image_list in all_values_per_image for value in image_list]
            # keep track of the indices - all_indices is a list of indices, the images to which each color value belongs
            all_indices = [index for index, sublist in enumerate(all_values_per_image) for item in sublist]

            _bprint(print_steps, "Clustering " + str(len(all_values)) + " colors...")
            clustered_values = _cluster(all_values, **cluster_args, show_color_scatter=show, show_color_centroids=show,
                                        input_colorspace=colorspace, return_values_as_centroids=True,
                                        print_steps=print_steps,show=show)

            def group_values_by_indices(values, indices):
                groups = {}
                for value, index in zip(values, indices):
                    if index not in groups:
                        groups[index] = []
                    groups[index].append(value)

                # Sort the dictionary by its keys and return the values
                return [groups[key] for key in sorted(groups)]

            # clustered_values_imgs regroups the clustered values back to their original images using the image indices we saved
            clustered_values_imgs = group_values_by_indices(clustered_values, all_indices)
            # we then create a new patch_or_pixel_data that only replaces old values with the new clustered values
            group_ppds = [(ppd[0], clustered_values_imgs[index], ppd[2]) for index, ppd in
                                   enumerate(group_ppds)]
            ##
            # I think indices are screwed up when you do this?
            updated_patch_or_pixel_data = updated_patch_or_pixel_data + group_ppds
            updated_img_ids = updated_img_ids + group_img_ids
        patch_or_pixel_data = updated_patch_or_pixel_data
        img_ids = updated_img_ids

    # cluster at the level of individual images
    elif cluster_individually:
        _bprint(print_steps, "Clustering individually")
        for index, ppd in enumerate(patch_or_pixel_data): # for each image (the patches or pixels within it)
            
            # cluster patch_or_pixel_data_point[1] which is the pixel colors OR the patch colors
            #   and return the centroids
            clustered_values = _cluster(ppd[1], **cluster_args, show_color_scatter=show, show_color_centroids=show, input_colorspace=colorspace, return_values_as_centroids=True,show=show)
            # reassign the centroids to the data
            ppd = (ppd[0], clustered_values, ppd[2])
            # put back into the list
            patch_or_pixel_data[index] = ppd

    else:  # otherwise we are grouping the ids (default)
        _bprint(print_steps, "Group clustering all IDs together...")

        # get color values per image (a list of lists, each sublist containing RGB color values each of which has a length of 3)
        all_values_per_image = [ppd[1] for ppd in patch_or_pixel_data]

        # combine the image sublists - all_values is a list of color values
        all_values = [value for image_list in all_values_per_image for value in image_list]
        # keep track of the indices - all_indices is a list of indices, the images to which each color value belongs
        all_indices = [index for index, sublist in enumerate(all_values_per_image) for item in sublist]

        _bprint(print_steps, "Clustering " + str(len(all_values)) + " colors...")
        clustered_values = _cluster(all_values, **cluster_args, show_color_scatter=show, show_color_centroids=True, input_colorspace=colorspace, return_values_as_centroids=True,print_steps=print_steps,show=show)

        def group_values_by_indices(values, indices):
            groups = {}
            for value, index in zip(values, indices):
                if index not in groups:
                    groups[index] = []
                groups[index].append(value)

            # Sort the dictionary by its keys and return the values
            return [groups[key] for key in sorted(groups)]

        # clustered_values_imgs regroups the clustered values back to their original images using the image indices we saved
        clustered_values_imgs = group_values_by_indices(clustered_values, all_indices)
        # we then create a new patch_or_pixel_data that only replaces old values with the new clustered values
        patch_or_pixel_data = [(ppd[0],clustered_values_imgs[index],ppd[2]) for index, ppd in enumerate(patch_or_pixel_data)]

    # reconstruct the images
    # patch_or_pixel_data is a list of tuples
    #   each tuple contains: a list of locations of the pixels and patches as its first element
    #                        a list of the values of the pixels or patches as its second element
    #                        the ndarray of image shapes
    patterns_to_show = []
    for i, ppd in enumerate(patch_or_pixel_data):
        img = _reconstructImgFromPPD(ppd,is_patch_data=by_patches,input_colorspace=colorspace)
        img = _format(img, in_format=colorspace,out_format="rgb",alpha=True)

        if i < 18:
            patterns_to_show.append(np.copy(img))

        img = _blackBgToTransparent(img)

        # setup write target
        write_target = data_folder + "/patterns/" + img_ids[i] + "_pattern.png"
        # if writing to subfolder
        if write_subfolder != "":
            if not os.path.exists(data_folder + "/patterns/" + write_subfolder):
                os.mkdir(data_folder + "/patterns/" + write_subfolder)
            # change write target to subfolder
            write_target = data_folder + "/patterns/" + write_subfolder + "/" + img_ids[i] + "_pattern.png"

        cv2.imwrite(write_target, img)
        if(print_details): print("Wrote to " + write_target)

        print(i)
    _showImages(show,patterns_to_show,maintitle="Final Patterns")

#def getPatchesAndPatchPixelMeans(img):
#    patch_img = _imgToColorPatches(img)

# given an image return a tuple containing pixel coords, pixel values, and the shape of the image
def getPixelCoordsAndPixels(img):
    img = _blackBgToTransparent(img) # turn this off, this is related to the same call at the end of clusterimgs
    pixels = []
    pixel_coords = []
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]): # for each pixel in the image
            pixel = img[i][j]
            if not (pixel[3] == 0): # if pixel is not transparent
                p = np.array([pixel[0],pixel[1],pixel[2]])

                pixels.append(p) # append pixel value
                pixel_coords.append((i, j)) # append coordinate of pixel in img
    return (pixel_coords,pixels,img.shape)