from bigcrittercolor.helpers import _clusterByImgFeatures, _getIDsInFolder, _readBCCImgs
import random
from bigcrittercolor.helpers.imgtransforms import _verticalizeImg

def test_clusterByImgFeatures_vgg16():
    folder = "D:/dfly_appr_expr/appr1"
    ids = _getIDsInFolder(folder + "/masks")
    random.seed(30)
    ids = random.sample(ids, 300)
    imgs = _readBCCImgs(ids, type="mask", data_folder=folder)
    _clusterByImgFeatures(imgs, feature_extract_cnn="vgg16")

def test_clusterByImgFeatures_resnet18():
    folder = "D:/dfly_appr_expr/appr1"
    ids = _getIDsInFolder(folder + "/masks")
    random.seed(30)
    ids = random.sample(ids,300)
    imgs = _readBCCImgs(ids, type="mask", data_folder=folder)
    _clusterByImgFeatures(imgs,feature_extract_cnn="resnet18")

def test_clusterByImgFeatures_inceptionv3():
    folder = "D:/dfly_appr_expr/appr1"
    ids = _getIDsInFolder(folder + "/masks")
    random.seed(30)
    ids = random.sample(ids, 300)
    imgs = _readBCCImgs(ids, type="mask", data_folder=folder)
    _clusterByImgFeatures(imgs, feature_extract_cnn="inceptionv3")

def test_clusterByImgFeatures_agglom():
    folder = "D:/dfly_appr_expr/appr1"
    ids = _getIDsInFolder(folder + "/masks")
    random.seed(30)
    ids = random.sample(ids, 300)
    imgs = _readBCCImgs(ids, type="mask", data_folder=folder)
    _clusterByImgFeatures(imgs, feature_extract_cnn="vgg16",cluster_algo="agglom")

def test_clusterByImgFeatures_kmeans():
    folder = "D:/dfly_appr_expr/appr1"
    ids = _getIDsInFolder(folder + "/masks")
    random.seed(30)
    ids = random.sample(ids, 300)
    imgs = _readBCCImgs(ids, type="mask", data_folder=folder)
    _clusterByImgFeatures(imgs, feature_extract_cnn="vgg16",cluster_algo="kmeans")

def test_clusterByImgFeatures_fulldisplay():
    folder = "D:/dfly_appr_expr/appr1"
    ids = _getIDsInFolder(folder + "/masks")
    random.seed(30)
    ids = random.sample(ids, 300)
    imgs = _readBCCImgs(ids, type="mask", data_folder=folder)
    _clusterByImgFeatures(imgs, full_display_ids=ids, full_display_data_folder=folder)

def test_clusterByImgFeatures_dbscan():
    folder = "D:/bcc/dfly_appr_expr/appr1"
    ids = _getIDsInFolder(folder + "/masks")
    random.seed(30)
    ids = random.sample(ids, 300)
    imgs = _readBCCImgs(ids, type="mask", data_folder=folder)
    imgs = [_verticalizeImg(img) for img in imgs]
    _clusterByImgFeatures(imgs, cluster_algo="dbscan", full_display_ids=ids, full_display_data_folder=folder)

def test_clusterByImgFeatures_fuzzy_gaussian():
    folder = "D:/bcc/dfly_appr_expr/appr1"
    ids = _getIDsInFolder(folder + "/masks")
    random.seed(30)
    ids = random.sample(ids, 300)
    imgs = _readBCCImgs(ids, type="mask", data_folder=folder)
    _clusterByImgFeatures(imgs, cluster_n=4, cluster_algo="gaussian_mixture", fuzzy_probs_threshold=0.7, full_display_ids=ids, full_display_data_folder=folder)

test_clusterByImgFeatures_dbscan()