import pandas as pd
from numpy import unique
from sklearn.cluster import KMeans, OPTICS, SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
import numpy as np
import cv2
from bigcrittercolor.helpers import _showImages
# from yellowbrick.cluster import KElbowVisualizer
# from kneed import DataGenerator, KneeLocator
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
import colorsys
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import color
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.cluster import hierarchy
from bigcrittercolor.helpers import _getIDsInFolder
from bigcrittercolor.helpers.clustering import _cluster

def clusterColorsToPatterns2(img_ids=None, preclustered = False, group_cluster_records_col = None, group_cluster_raw_ids = True,
                    by_contours=False, contours_erode_size=0, contours_dilate_mult = 2, min_contour_pixel_area=10,
                    cluster_algo="gaussian_mixture", cluster_n=None, cluster_params_dict='',
                    colorspace = None, scale = False, use_positions=False, downweight_axis=None, upweight_axis=None,
                    vert_resize = 100,
                    gaussian_blur= False, median_blur = False, bilat_blur = False, blur_size = 15, blur_sigma = 150,
                    print_steps=True, print_details=False, preclust_read_subfolder = "", write_subfolder= "",
                    show=False, show_indv=False, data_folder="../.."):

    """ Discretize (AKA quantize or recolorize) continuously shaded organismal segments to discrete patterns

        Args:
            img_ids (list): the imageIDs (image names) to use segments from
            preclustered (bool): specify whether you want to cluster preclustered data, applying the second step in a typical 2-part clustering run
            group_cluster_records_col (str): the column in records to group images before clustering i.e "species"
                can also be "speciesMorph" which gets merged from inferences
            group_cluster_raw_ids (bool): whether to group cluster the provided ids as whole, rather than splitting them into groups with the group_cluster_records_col

            by_contours (bool): whether to cluster by contours instead of by pixels
            contours_erode_size (int): the size of the kernel with which to erode contours, no erosion if 0
            contours_dilate_mult (float): the amount to dilate contours
            min_contour_pixel_area (int): the minimum number of pixels in a contour to include it as a pattern element in clustering

            cluster_algo (str): the type of clustering to perform, either "kmeans" or "optics" for now
            cluster_n (int): the number of clusters, if None the cluster_n are chosen using the nclust_metric
            cluster_params_dict (dictionary):
            colorspace (str): the colorspace to cluster in, None uses rgb, "hls" uses HLS, and "lab" uses CIELAB
            scale (bool): whether to scale the pixel data together before clustering
            use_positions (bool): whether to cluster using pixel positions within the image in addition to pixel values
                This means that pixels that are closer together will be more likely to be assigned to the same cluster

            downweight_axis: the index of the color axis (such as R,G,B or H,S,L) to downweight
            upweight_axis: the index of the color axis (such as R,G,B or H,S,L) to upweight
            vert_resize: the number of pixels to resize to in height before pixels are extracted and clustered, keeping the image aspect ratio intact

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

    if img_ids is None:
        img_ids = _getIDsInFolder(data_folder + "/segments")
        if preclustered:
            img_ids = _getIDsInFolder(data_folder + "/patterns")

    img_ids = img_ids.copy()
    populated_ids = []

    # gather either contour data or pixel data for every image
    contour_or_pixel_data = []
    if (print_steps):
        if preclustered:
            print("Estimated time " + str(len(img_ids)/30) + " minutes at default cluster mode, metric, and vert resize")
        else:
            print("Estimated time " + str((len(img_ids)/30)/3) + " minutes at default cluster mode, metric, and vert resize")

    # load BGR images
    if print_steps: print("Started gathering pixel or contour data for each image")
    for id in img_ids:
        if not preclustered:
            img = cv2.imread(data_folder + "/segments/" + id + "_segment.png",cv2.IMREAD_UNCHANGED)
        else:
            img = cv2.imread(data_folder + "/patterns/" + preclust_read_subfolder + "/" + id + "_segment.png", cv2.IMREAD_UNCHANGED) #_pattern.ong
            if(show_indv):
                print("Unique colors in preclustered image:")
                pixels = img.reshape((-1, img.shape[-1]))
                print(np.unique(pixels,axis=0))

        _showImages(show_indv,[img],"Segment")

        if print_details: print("Loaded segment for ID " + id)
        if img is None:
            if print_details: print("Image is empty - skipping")
            continue

        # apply bilateral blur
        if bilat_blur:
            alpha = img[:, :, 3]
            img = img[:, :, :3]
            img = cv2.bilateralFilter(img, d=blur_size, sigmaColor=blur_sigma, sigmaSpace=blur_sigma)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            img[:, :, 3] = alpha
            _showImages(show_indv,[img],['Blurred Image'])

        # apply gaussian blur
        if gaussian_blur:
            img = cv2.GaussianBlur(img, ksize=(blur_size, blur_size), sigmaX = blur_sigma, sigmaY= blur_sigma)
            _showImages(show_indv, [img], ['Blurred Image'])

        # apply gaussian blur
        if median_blur:
            img = cv2.medianBlur(img, blur_size)
            _showImages(show_indv, [img], ['Blurred Image'])

        # convert to hls or lab
        if colorspace == "hls":
            alpha = img[:, :, 3] # save alpha
            img = img[:, :, :3] # remove alpha
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS_FULL)
            img = np.dstack((img, alpha))
        elif colorspace == "lab":
            alpha = img[:, :, 3]  # save alpha
            img = img[:, :, :3]  # remove alpha
            img = color.rgb2lab(img,illuminant="E")
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            img = np.dstack((img, alpha)) # return alpha (necessary for excluding pixels)
            if(show_indv):
                print("Unique colors in preclustered image converted to LAB:")
                pixels = img.reshape((-1, img.shape[-1]))
                print(np.unique(pixels,axis=0))

        # resize
        if vert_resize is not None:
            dim_mult = vert_resize / img.shape[0]
            img = cv2.resize(img, dsize=(int(img.shape[1] * dim_mult),vert_resize),interpolation = cv2.INTER_NEAREST)
        if show_indv:
            print("NA")
            #plotPixels(img[:, :, :3],colorspace)

        if by_contours:
            contour_or_pixel_data.append(getContoursAndContourPixelMeans(img,contours_erode_size,contours_dilate_mult,min_contour_pixel_area,show))
            if print_details: print("Gathered contour data")
        else:
            contour_or_pixel_data.append(getPixelsAndPixelCoords(img))
            if print_details: print("Gathered pixel data")
        populated_ids.append(id)

    # cluster by group
    if group_cluster_records_col is not None:
        if print_steps: print("Started group clustering by a records column (probably species or clade)")
        # load records and keep those with matching ids
        records = pd.read_csv("data/inatdragonflyusa_records.csv")
        records = records[records["imageID"].isin(img_ids),:]

        # loop through unique groups and update contour data using clustered centroids
        unique_groups = unique(records[group_cluster_records_col])
        for g in unique_groups:
            # group_ids = records.query(group_cluster_records_col + '==' + g)["imageID"]

            # get indices for the group
            group_indices = records.index[records[group_cluster_records_col]==g].tolist()

            # get data for the group
            group_data = contour_or_pixel_data[group_indices]

            # get 2nd element of each tuple, the pixel means
            group_contour_means = [c[1] for c in group_data]

            cluster_eps = 0
            cluster_min_samples = 0
            nclust_metric = None

            # cluster and get new values from cluster centroids
            #clustered_values = getClusterCentroids(group_contour_means,cluster_algo,cluster_n,cluster_eps,cluster_min_samples,scale,use_positions,downweight_axis,upweight_axis,preclustered,nclust_metric,img,colorspace,show)

            # replaced old values in group data with new values
            for index, d in enumerate(group_data):
                group_data[index] = tuple(d[0] + clustered_values[index] + d[2])

            # add group data back to main data
            contour_or_pixel_data[group_indices] = group_data

    elif group_cluster_raw_ids:
        if print_steps: print("Group clustering all IDs together")
        # get pixels or contours then take mean (only affects contours)
        group_means = [cpd[1] for cpd in contour_or_pixel_data]
        group_means = sum(group_means, [])

        # get coords
        group_coords = [cpd[0] for cpd in contour_or_pixel_data]

        print("Len group means " + str(len(group_means)))
        cluster_eps = 0
        cluster_min_samples = 0
        nclust_metric = None
        clustered_values = _cluster(group_means, algo=cluster_algo, n=cluster_n, return_values_as_centroids=True)
        print(clustered_values)
        clustered_values = clustered_values / 255
        print("Len clust vals " + str(len(clustered_values)))
        i = 0
        for index, cpd in enumerate(contour_or_pixel_data):
            cpd = list(cpd)
            #print("Len cpd " + str(len(cpd[1])))
            # set cpd mean values to clustered values
            n_values = len(cpd[1])
            cpd[1] = clustered_values[i:(n_values + i)]
            i = i + n_values
            cpd = tuple(cpd)
            #print("Len cpd 2 " + str(len(cpd[1])))
            contour_or_pixel_data[index] = cpd

    # if not grouping, just cluster at the level of individual images
    else:
        if print_steps: print("Clustering individually")
        for index, cpd in enumerate(contour_or_pixel_data):
            cpd = list(cpd)
            if print_details: print("Clustering next image")
            cluster_eps = 0
            cluster_min_samples = 0
            nclust_metric = None
            clustered_values =  _cluster(cpd[1], algo="gaussian_mixture", n=5, return_values_as_centroids=True)
            #clustered_values = getClusterCentroids(cpd[1], cpd[0], cluster_algo, cluster_n, cluster_eps, cluster_min_samples,scale,use_positions,downweight_axis,upweight_axis,preclustered,nclust_metric,img,colorspace,show)
            cpd[1] = clustered_values
            cpd = tuple(cpd)
            contour_or_pixel_data[index] = cpd

    # loop over cluster-centered data
    i = 0
    for cpd in contour_or_pixel_data:
        img = np.zeros(shape=cpd[2])
        if by_contours:
            contours = cpd[0]
            contour_means = cpd[1]
            # draw contours with mean colors
            for j in range(len(contours)):
                # Create a mask image that contains the contour filled in
                img = cv2.drawContours(img, contours, j, color=contour_means[j], thickness=-1)
        else:
            # messy code to reassign pixel values, unsure how to do this properly with numpy
            pixel_coords = cpd[0]
            pixel_vals = cpd[1]

            for index, p in enumerate(pixel_vals):
                #p = np.array(colorsys.hls_to_rgb(p[0]/255,p[1]/255,p[2]/255))
                p = np.array((p[0], p[1], p[2]))
                p = np.append(p,255) # add opaque alpha
                #print(p)
                img[pixel_coords[index]] = p

        #img = img / 255

        if colorspace == "hls":
            # convert back to RGB from HLS
            alpha = img[:, :, 3]  # save alpha
            img = img[:, :, :3]  # remove alpha
            img = img.astype(np.float32)
            img = cv2.cvtColor(img, cv2.COLOR_HLS2BGR_FULL)
            img = np.dstack((img, alpha))
        elif colorspace == "lab":
            alpha = img[:, :, 3]
            img = img[:, :, :3]
            img = img.astype(np.float32)
            img = color.lab2rgb(img,illuminant="E")
            img = np.dstack((img, alpha))

        #alpha = img[:, :, 3]
        #img = img[:, :, :3]
        #img = img.astype(np.float32)
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        #img[:, :, 3] = alpha

        if show_indv:
            cv2.imshow("i", img)
            cv2.waitKey(0)

        if colorspace != "rgb":
            img = img * 255

        # reshape to original dims
        #img = img.reshape(cpd[2])

        #_showImages(show,[img],['Discretized'])
        if write_subfolder != "":
            if not os.path.exists(data_folder + "/patterns/" + write_subfolder):
                os.mkdir(data_folder + "/patterns/" + write_subfolder)
            write_subfolder = write_subfolder + "/"
        write_target = data_folder + "/patterns/" + write_subfolder + populated_ids[i] + "_pattern.png"
        cv2.imwrite(write_target, img)
        if(print_details): print("Wrote to " + write_target)

        print(i)
        i = i + 1

# key function that is given an image (png RGBA segment) and returns a tuple three lists
#   the first list containing contours (lists of contour points),
#   the second list containing pixel means for each contour
#   the third list containing image dimensions for reconstruction
def getContoursAndContourPixelMeans(img,contours_erode_size,contours_dilate_mult,min_contour_pixel_area,show):
    show=True
    # convert img to grey
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # get threshold image to use when finding contours
    thresh_img = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

    cv2.imshow("0",thresh_img)
    cv2.waitKey(0)

    # erode
    if contours_erode_size != 0:
        kernel = np.ones((contours_erode_size, contours_erode_size), np.uint8)
        thresh_img = cv2.erode(thresh_img, kernel)
        thresh_img = cv2.dilate(thresh_img, kernel * contours_dilate_mult)

    # find contours
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # keep contours above the contour area threshold
    new_conts = list()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_contour_pixel_area:
            new_conts.append(cnt)
    # append these new contours to all contours
    contours = tuple(new_conts)

    # initialize empty list to hold the mean pixel values of each contour
    contour_means = []

    # for each list of contour points...
    for i in range(len(contours)):
        # create a mask image that contains the contour filled in
        cimg = np.zeros_like(img)
        cv2.drawContours(cimg, contours, i, color=255, thickness=-1)

        # access the image pixels and create a 1D numpy array
        pts = np.where(cimg == 255)

        # convert image
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        #if color_fun is not None: #cv2.COLOR_RGB2LAB, RGB2HSV
        #    img = cv2.cvtColor(img, color_fun)



        # add the mean values of these pixels to contour_means
        contour_means.append(np.mean(img[pts[0], pts[1]], axis=0))  # img_lab

    # start contour image as containing the whole segment
    ret,thresh_img_mask = cv2.threshold(img_grey,1,255,cv2.THRESH_BINARY)
    #thresh_img_mask = cv2.cvtColor(thresh_img_mask, cv2.COLOR_BGR2GRAY)

    # get the contour of the whole segment
    primary_contours, hierarchy = cv2.findContours(thresh_img_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    def random_color():
        """Generate a random color."""
        return (int(np.random.randint(0, 255)), int(np.random.randint(0, 255)), int(np.random.randint(0, 255)))

    # Create a blank image to draw the contours
    output = np.zeros_like(img, dtype=np.uint8)

    # Draw all the contours
    for contour in contours:
        cv2.drawContours(output, [contour], -1, random_color(),
                         2)  # Here, (255, 0, 0) is the color and 2 is the thickness.

    # Show the image with contours
    cv2.imshow('Contours', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # create a mask image that contains the contour filled in
    cont_img = np.zeros_like(img)
    for i in range(len(primary_contours)):
        cv2.drawContours(cont_img, primary_contours, i, color=[255,255,255,255], thickness=-1)

    _showImages(show, [cont_img], ["Primary contour"])

    # for every NON-PRIMARY MASK contour
    # fill in contour image
    for i in range(len(contours)):
        # create a mask image that contains the contour filled in
        cv2.drawContours(cont_img, contours, i, color=[0, 0, 0, 0], thickness=-1)

    _showImages(show, [cont_img], ["Primary contour backfilled with other contours"])

    # Access the image pixels and create a 1D numpy array then add to list
    primary_contour_points = np.where(cont_img != 0)
    primary_cont_mean = np.mean(img[primary_contour_points[0], primary_contour_points[1]], axis=0)

    contours = list(contours)
    # add primary contour mean to contour means
    for i in range(len(primary_contours)):
        contours.insert(0,primary_contours[i])
        contour_means.insert(0,primary_cont_mean)

    return (contours,contour_means,img.shape)

from bigcrittercolor.helpers.image import _blackBgToTransparent

# return a tuple containing pixel coords, pixel values, and the shape of the image
def getPixelsAndPixelCoords(img):
    img = _blackBgToTransparent(img)
    pixels = []
    pixel_coords = []
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            pixel = img[i][j]
            if not (pixel[3] == 0):
                p = np.array([pixel[0],pixel[1],pixel[2]])
                #print(p)
                #if color_fun == cv2.COLOR_RGB2HLS:
                    #p[0] = p[0] / 180
                    #p[1] = p[1] / 255
                    #p[2] = p[2] / 255
                #else:
                    #p = p / 255
                #p = np.array(pixel[0])

                pixels.append(p) # append 0-1
                pixel_coords.append((i, j))
    return (pixel_coords,pixels,img.shape)

def hls_to_rgb_mat(hls_matrix):
    # hls_matrix should be a 2D matrix with each row containing
    # the H, L, and S values for a pixel
    # initialize a 2D matrix for RGB values
    rgb_matrix = [[0 for j in range(len(hls_matrix[0]))] for i in range(len(hls_matrix))]

    for i in range(len(hls_matrix)):
        for j in range(len(hls_matrix[0])):
            # convert HLS values to RGB values
            r, g, b = colorsys.hls_to_rgb(hls_matrix[i][j][0], hls_matrix[i][j][1], hls_matrix[i][j][2])
            # assign RGB values to the corresponding pixel
            rgb_matrix[i][j] = (r, g, b)
    return rgb_matrix