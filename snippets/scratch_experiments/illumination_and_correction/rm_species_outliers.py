import cv2
import numpy as np
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Function to extract RGB features
def extract_rgb_features(image):
    mean_r = np.mean(image[:, :, 2])
    std_r = np.std(image[:, :, 2])
    mean_g = np.mean(image[:, :, 1])
    std_g = np.std(image[:, :, 1])
    mean_b = np.mean(image[:, :, 0])
    std_b = np.std(image[:, :, 0])
    return [mean_r, std_r, mean_g, std_g, mean_b, std_b]

from bigcrittercolor.helpers import _readBCCImgs, _getBCCIDs
from bigcrittercolor.helpers import _showImages

ids = _getBCCIDs(type="segment",records_filter_dict={'species':"Erpetogomphus designatus"},data_folder="D:/bcc/ringtails")
images = _readBCCImgs(img_ids=ids,type="segment",data_folder="D:/bcc/ringtails")

features = [extract_rgb_features(img) for img in images]
features = np.array(features)

# Normalize the features
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

# Identify outliers using Isolation Forest
iso_forest = IsolationForest(contamination=0.10)  # Adjust contamination based on your needs
outliers = iso_forest.fit_predict(normalized_features)

# Separate inliers and outliers
inliers = [images[i] for i in range(len(outliers)) if outliers[i] == 1]
outliers = [images[i] for i in range(len(outliers)) if outliers[i] == -1]


print("Outlier images:", len(outliers))
print("Inlier images:", len(inliers))

_showImages(True,images=outliers,sample_n=18)
_showImages(True,images=inliers,sample_n=18)