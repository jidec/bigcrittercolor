import os
import cv2
import numpy as np
from collections import Counter

from bigcrittercolor.helpers import (
    _showImages, _bprint, _readBCCImgs, _clusterByImgFeatures, _getBCCIDs
)

def clusterImagesDirect(img_ids=None, sample_n=None,
                        downscale_max_side=512,
                        blur_ksize=0,                      # set >0 to lightly blur (e.g., 5 or 7)
                        feature_extractor="resnet18",
                        cluster_params_dict=None,
                        show=True, show_save=True, print_steps=True,
                        data_folder="", save_folder=None):
    """
    Cluster the ORIGINAL images (no masking) via CNN features and visualize clusters.

    Args:
        img_ids: list of image IDs. If None, loads from data_folder via _getBCCIDs(type="image").
        sample_n: optional subsample of IDs for quick runs.
        downscale_max_side: resize so max(h, w) <= this (speeds up feature extraction).
        blur_ksize: optional Gaussian blur kernel (odd int). 0 = no blur.
        feature_extractor: passed to _clusterByImgFeatures (e.g., 'resnet18', 'vgg16', 'inceptionv3').
        cluster_params_dict: dict for clustering (algo, pca_n, n, scale, etc.).
        show/show_save/print_steps: display & logging toggles.
        data_folder: BCC project folder.
        save_folder: where to save grids (defaults to data_folder/plots).

    Returns:
        labels: np.array of cluster labels aligned with kept_ids
        kept_ids: list of image IDs actually clustered (skips unreadable images)
    """
    if cluster_params_dict is None:
        cluster_params_dict = {
            'algo': "gaussian_mixture",
            'pca_n': 5,
            'n': 8,              # tweak for your dataset size
            'scale': "standard",
            'show_pca_tsne': True,
            'show_clusters': True
        }

    if save_folder is None and data_folder:
        save_folder = os.path.join(data_folder, "plots")

    _bprint(print_steps, "Clustering RAW images (no subject removal)...")

    # fetch IDs if not provided
    if img_ids is None:
        img_ids = _getBCCIDs(type="image", data_folder=data_folder)
    img_ids = list(img_ids)

    # optional subsample
    if sample_n is not None and len(img_ids) > sample_n:
        rng = np.random.default_rng()
        img_ids = list(rng.choice(img_ids, size=sample_n, replace=False))

    # load images
    _bprint(print_steps, f"Reading {len(img_ids)} images...")
    imgs = _readBCCImgs(img_ids=img_ids, type="image", data_folder=data_folder)

    # _readBCCImgs returns a list for a list of IDs; guard for single-image edge case
    if not isinstance(imgs, list):
        imgs = [imgs]
    assert len(imgs) == len(img_ids)

    kept_imgs, kept_ids, failed = [], [], []

    # preprocess (optional blur + downscale)
    for img_id, im in zip(img_ids, imgs):
        if im is None:
            failed.append((img_id, "unreadable"))
            continue

        # blur (optional)
        if blur_ksize and blur_ksize > 0:
            k = int(blur_ksize)
            if k % 2 == 0:
                k += 1
            im = cv2.GaussianBlur(im, (k, k), 0)

        # downscale to cap max side
        if downscale_max_side and downscale_max_side > 0:
            h, w = im.shape[:2]
            max_side = max(h, w)
            if max_side > downscale_max_side:
                scale = downscale_max_side / float(max_side)
                im = cv2.resize(im, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        kept_imgs.append(im)
        kept_ids.append(img_id)

    if len(kept_imgs) == 0:
        _bprint(True, "No valid images to cluster. Aborting.")
        return np.array([]), []

    if failed:
        _bprint(True, f"Skipped {len(failed)} images (e.g., {failed[:5]})")

    _bprint(print_steps, f"Clustering {len(kept_imgs)} images with {feature_extractor} features...")

    # cluster and visualize via your helper
    labels = _clusterByImgFeatures(
        kept_imgs,
        feature_extractor=feature_extractor,
        full_display_ids=None,               # directly display our preprocessed images
        data_folder=data_folder,
        print_steps=print_steps,
        cluster_params_dict=cluster_params_dict,
        show=show,
        show_save=show_save
    )

    # per-cluster grids
    uniq = sorted(set(labels))
    counts = Counter(labels)
    _bprint(print_steps, f"Found {len(uniq)} clusters: " + ", ".join(f"{c} (n={counts[c]})" for c in uniq))

    for c in uniq:
        imgs_c = [kept_imgs[i] for i, lab in enumerate(labels) if lab == c]
        title = f"Image Cluster {c} (n={len(imgs_c)})"
        _showImages(True, imgs_c, sample_n=min(25, len(imgs_c)), maintitle=title, save_folder=save_folder)

    _bprint(print_steps, f"Done. Total clustered images: {len(kept_ids)}")
    return np.array(labels), kept_ids

clusterImagesDirect(data_folder="D:/bcc/chloros")