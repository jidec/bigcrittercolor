from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, KMeans, SpectralClustering
import torch
from torchvision import models, transforms
import numpy as np
from bigcrittercolor.helpers import _showImages, _bprint, makeCollage
from bigcrittercolor.helpers._readBCCImgs import _readBCCImgs
from bigcrittercolor.helpers.clustering import _cluster, _findNClusters
from torch import nn
import random
from PIL import Image
from scipy.cluster import hierarchy
from matplotlib import pyplot as plt
import cv2
from bigcrittercolor.helpers.image import _padImgToSize,_resizeImgToTotalDim

# helper that takes a list of numpy arr images and returns labels clustered by AffinityProp on features
#   extracted by a pretrained VGG16 model
def _clusterByImgFeatures(imgs, feature_extractor="resnet18", cluster_algo="kmeans", cluster_n=3,
                            cluster_params_dict={'eps':10,'min_samples':3}, fuzzy_probs_threshold=None,
                            pad_imgs =True,
                            full_display_ids=None, full_display_data_folder=None,
                            print_steps=True, show=True, show_n=18):

    if pad_imgs:
        imgs = [_resizeImgToTotalDim(img,800) for img in imgs]
        imgs = [_padImgToSize._padImgToSize(img,(900,900)) for img in imgs]
    # load model
    _bprint(print_steps,"Loading pretrained feature extractor...")
    match feature_extractor:
        case "vgg16":
            model = models.vgg16(pretrained=True)
            new_model = FeatureExtractor(model)

            # Change the device to GPU
            device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
            new_model = new_model.to(device)

            # function to transform the image, so it becomes readable with the model
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop(512),
                transforms.Resize(448),
                transforms.ToTensor()
            ])
            # a list containing the features for each mask (i.e. each index is one mask)
            features = []

            for img in imgs:
                # transform the image
                img = transform(img)

                # reshape the image. PyTorch model reads 4-dimensional tensor
                # [batch_size, channels, width, height]
                img = img.reshape(1, 3, 448, 448)
                img = img.to(device)
                # we only extract features, so we don't need gradient
                with torch.no_grad():
                    # extract the feature from the image
                    feature = new_model(img)
                    # convert to NumPy Array, reshape it, and save it to features variable
                    features.append(feature.cpu().detach().numpy().reshape(-1))

        case "resnet18":
            model = models.resnet18(pretrained=True)
            model = torch.nn.Sequential(*(list(model.children())[:-1]))
            model.eval()

            # Define the transformation
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            features = []
            for img in imgs:
                image = Image.fromarray(img)
                image = transform(image)
                image = image.unsqueeze(0)  # Add batch dimension
                new_features = model(image)
                features.append(new_features.cpu().detach().numpy().reshape(-1))
        case "inceptionv3":
            # Load the pre-trained model
            inception = models.inception_v3(pretrained=True)
            # Remove the last layer (classification layer)
            # We'll use the features from the auxiliary net
            # Note that aux_logits must be True to get aux_out
            inception.AuxLogits.fc = torch.nn.Identity()
            # We'll also remove the final fully connected layer from the main output
            inception.fc = torch.nn.Identity()
            # Put the model into evaluation mode
            inception.eval()
            # Define the transformation
            transform = transforms.Compose([
                transforms.Resize(299),  # Inception v3 expects 299x299 images
                transforms.CenterCrop(299),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            features = []
            for img in imgs:
                image = Image.fromarray(img)
                image = transform(image)
                image = image.unsqueeze(0)  # Add batch dimension
                new_features = inception(image)
                features.append(new_features.cpu().detach().numpy().reshape(-1))

    features = np.array(features)

    # cluster
    _bprint(print_steps,"Clustering mask features...")

    if cluster_n is None:
        cluster_n = _findNClusters(features, show=show)

    if fuzzy_probs_threshold is not None:
        probs = _cluster(features,params_dict=cluster_params_dict,algo=cluster_algo,n=cluster_n,return_fuzzy_probs=True)
        # Get the indices of the maximum probabilities
        max_prob_indices = np.argmax(probs, axis=1)
        # Get the values of the maximum probabilities
        max_probs = np.max(probs, axis=1)
        # Assign labels based on the threshold
        labels = np.where(max_probs >= fuzzy_probs_threshold, max_prob_indices, -1)

    else:
        labels = _cluster(features,algo=cluster_algo,n=cluster_n, **cluster_params_dict)

    if show:
        # for each cluster label
        for cluster_label in list(set(labels)):
            cluster_imgs = []
            cluster_ids = []
            for index, label in enumerate(labels):
                # get all masks matching that label
                if label == cluster_label:
                    cluster_imgs.append(imgs[index])
                    # if ids for full display are supplied, hold on to them
                    if full_display_ids is not None:
                        cluster_ids.append(full_display_ids[index])

            n_imgs_in_cluster = len(cluster_imgs) # save n imgs in cluster
            # reduce number of images as there will often be hundreds or thousands
            if len(cluster_imgs) > show_n:
                rand_indices = random.sample(range(len(cluster_imgs)), show_n)
                cluster_imgs = [cluster_imgs[i] for i in rand_indices]
                # if using full display keep track of ids
                if full_display_ids is not None:
                    cluster_ids = [cluster_ids[i] for i in rand_indices]

            # if using full display load in corresponding images and masks and apply them, then collage segments in
            if full_display_ids is not None:
                images = _readBCCImgs(cluster_ids,type="img",data_folder=full_display_data_folder)
                masks = _readBCCImgs(cluster_ids, type="mask", data_folder=full_display_data_folder)

                #for index, mask in enumerate(masks):
                #    print(cluster_ids[index])
                #    parent_img = images[index]
                #    segment = cv2.bitwise_and(parent_img, parent_img, mask=cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY).astype(np.uint8))
                segments = [cv2.bitwise_and(parent_img, parent_img,
                            mask=cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY).astype(np.uint8)) for
                            mask, parent_img in zip(masks,images)]
                cluster_imgs = [makeCollage.makeCollage([img, segment], n_per_row=2) for img, segment in zip(cluster_imgs, segments)]

            # visualize them for the user
            _showImages(show, cluster_imgs, titles=None,maintitle= "Cluster " + str(cluster_label) + ", " + str(n_imgs_in_cluster) + " images")

    # return labels
    return labels

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
