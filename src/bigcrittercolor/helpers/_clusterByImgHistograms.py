import cv2
import random
from bigcrittercolor.helpers.clustering import _cluster
from bigcrittercolor.helpers import _showImages

def _clusterByImgHistograms(imgs, cluster_params_dict,show=True,show_save=False,data_folder=''):
    histograms = []
    n_hist_bins = [32,32,32]

    # Compute histograms for each image, ignoring pure black pixels
    for img in imgs:
        # Convert image to HSV
        #hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Create mask to ignore pure black pixels
        mask = cv2.inRange(img, (1, 1, 1), (255, 255, 255))

        # Compute histogram for the image
        #hist = cv2.calcHist([img], [0, 1, 2], mask, n_hist_bins, [0, 180, 0, 256, 0, 256])
        hist = cv2.calcHist([img], [0, 1, 2], mask, n_hist_bins, [0, 256, 0, 256, 0, 256])

        hist = cv2.normalize(hist, hist).flatten()
        hist = hist.flatten()
        histograms.append(hist)

    # show is always true here because we have to show for user input
    #labels = _clusterByImgFeatures(images, feature_extractor="resnet18",full_display_data_folder="D:/bcc/ringtails",
    #                               print_steps=True, cluster_params_dict=hist_cluster_args_dict, show=True,
    #                               show_save=True)

    labels = _cluster(histograms,**cluster_params_dict)

    show_n = 30
    if show:
        # for each cluster label
        for cluster_label in list(set(labels)):
            cluster_imgs = []
            cluster_ids = []
            for index, label in enumerate(labels):
                # get all imgs matching that label
                if label == cluster_label:
                    cluster_imgs.append(imgs[index])

            n_imgs_in_cluster = len(cluster_imgs) # save n imgs in cluster
            # reduce number of images as there will often be hundreds or thousands
            if len(cluster_imgs) > show_n:
                cluster_imgs = random.sample(cluster_imgs,show_n)

            # visualize them for the user
            save_folder = None
            if show_save:
                save_folder = data_folder + "/plots"

            #temp_folder_path = data_folder + "/other/temp_filter_clusters/" + str(cluster_label)
            #if not os.path.exists(temp_folder_path):
            #    os.mkdir(temp_folder_path)
            #for i, img in enumerate(cluster_imgs):
            #    img_path = os.path.join(temp_folder_path, f"{i}.jpg")
            #    cv2.imwrite(img_path, img)

            _showImages(show, cluster_imgs, titles=None,maintitle= "Cluster " + str(cluster_label) + ", " + str(n_imgs_in_cluster) + " images",save_folder=save_folder)
    return labels
