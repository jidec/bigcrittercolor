from sklearn.cluster import AffinityPropagation
import torch
from torchvision import models, transforms
import numpy as np
from bigcrittercolor.helpers import _showImages, _bprint
from torch import nn
import random

# helper that takes a list of numpy arr images and returns labels clustered by AffinityProp on features
#   extracted by a pretrained VGG16 model
def _clusterByImgFeatures(imgs, feature_cnn="vgg16", clust_algo="affprop", affprop_pref=None,affprop_damping = 0.7, print_steps=True, show=False, show_n=18):

    # load model
    _bprint(print_steps,"Loading pretrained feature extractor...")
    model = models.vgg16(pretrained=True)
    #model = models.resnet18(pretrained=True)
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

    _bprint(print_steps,"Extracting features for each mask...")
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

    # convert to NumPy Array
    features = np.array(features)

    # cluster
    _bprint(print_steps,"Clustering mask features using affinity propagation...")
    affprop = AffinityPropagation(affinity="euclidean", preference= affprop_pref, damping=affprop_damping).fit(features)

    # get cluster labels
    labels = affprop.labels_

    if show:
        # for each cluster label
        for cluster_label in list(set(labels)):
            cluster_imgs = []
            for index, label in enumerate(labels):
                # get all masks matching that label
                if label == cluster_label:
                    cluster_imgs.append(imgs[index])

            # reduce number of images as there will often be hundreds or thousands
            if len(cluster_imgs) > 10:
                cluster_imgs = random.sample(cluster_imgs, show_n)

            # visualize them for the user
            _showImages(show, cluster_imgs, titles=None,maintitle=cluster_label)

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