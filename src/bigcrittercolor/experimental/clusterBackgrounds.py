import os
import cv2
import numpy as np
from collections import Counter

from bigcrittercolor.helpers import _showImages, _bprint, _readBCCImgs, _clusterByImgFeatures, _getBCCIDs

def clusterBackgrounds(img_ids=None, sample_n=None, batch_size=None,
                       mask_dilate_px=6,
                       fill_subject_with_bg_median=True,
                       blur_ksize=7,
                       downscale_max_side=512,
                       feature_extractor="resnet18",
                       cluster_params_dict=None,
                       show=True, show_save=True, print_steps=True,
                       data_folder="", save_folder=None):
    """
    Cluster the BACKGROUNDS of images (using the existing masks) and visualize clusters.

    Steps per image:
      - Read raw image + mask
      - Build a background-only thumbnail by inverting (optionally dilated) subject mask
      - Optionally fill the subject area with the median background color to avoid big black holes
      - Light blur + downscale (keeps habitat texture/color; reduces detail/memory)
      - Cluster background thumbs via CNN features using _clusterByImgFeatures
      - Visualize per-cluster grids with _showImages

    Args mirror your existing style; returns (labels, kept_ids) for convenience.
    """
    if cluster_params_dict is None:
        # Reasonable defaults; tweak n for your dataset size
        cluster_params_dict = {
            'algo': "gaussian_mixture",
            'pca_n': 5,
            'n': 8,
            'scale': "standard",
            'show_pca_tsne': True,   # if your _clusterByImgFeatures supports it
            'show_clusters': True
        }

    if save_folder is None and data_folder:
        save_folder = os.path.join(data_folder, "plots")

    _bprint(print_steps, "Clustering BACKGROUNDS (not subjects)...")

    # If no IDs provided, use whatever masks already exist
    if img_ids is None:
        img_ids = _getBCCIDs(type="mask", data_folder=data_folder)

    # Make list & optionally sample
    img_ids = list(img_ids)
    if sample_n is not None and len(img_ids) > sample_n:
        # NOTE: sampling is handled inside _readBCCImgs too, but we want to sample IDs consistently for masks+images
        rng = np.random.default_rng()
        img_ids = list(rng.choice(img_ids, size=sample_n, replace=False))

    # Batch if requested
    batches = [img_ids]
    if batch_size is not None:
        batches = [img_ids[i:i + batch_size] for i in range(0, len(img_ids), batch_size)]

    all_labels = []
    all_kept_ids = []

    for batch_num, batch_ids in enumerate(batches, 1):
        _bprint(print_steps, f"[Batch {batch_num}/{len(batches)}] Preparing backgrounds for {len(batch_ids)} IDs...")

        bg_thumbs = []
        kept_ids = []

        for idx, img_id in enumerate(batch_ids):
            if idx % 100 == 0:
                _bprint(print_steps, f"  {idx}/{len(batch_ids)}")

            # Read raw image & mask
            img = _readBCCImgs(img_id, type="image", data_folder=data_folder)
            mask = _readBCCImgs(img_id, type="mask", data_folder=data_folder)

            if img is None or mask is None:
                continue

            # Convert mask to grayscale and binarize
            if mask.ndim == 3:
                mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            else:
                mask_gray = mask.copy()

            if mask_gray is None or np.count_nonzero(mask_gray) == 0:
                # No subject mask; skip
                continue

            # Dilate the subject mask slightly to remove halo/edge pixels from BG
            subject = (mask_gray > 0).astype(np.uint8) * 255
            if mask_dilate_px and mask_dilate_px > 0:
                k = max(1, int(mask_dilate_px))
                kernel = np.ones((k, k), np.uint8)
                subject = cv2.dilate(subject, kernel, iterations=1)

            bg_mask = cv2.bitwise_not(subject)

            # Keep only background pixels
            bg = cv2.bitwise_and(img, img, mask=bg_mask)

            # Fill the subject region with median background color so black holes don't dominate features
            if fill_subject_with_bg_median:
                m = bg_mask > 0
                if m.any():
                    med_b = int(np.median(img[:, :, 0][m]))
                    med_g = int(np.median(img[:, :, 1][m]))
                    med_r = int(np.median(img[:, :, 2][m]))
                    fg = subject > 0
                    bg[fg] = (med_b, med_g, med_r)

            # Optional light blur (smooths noise; keeps broad habitat cues)
            if blur_ksize and blur_ksize > 0:
                k = int(blur_ksize)
                if k % 2 == 0:
                    k += 1
                bg = cv2.GaussianBlur(bg, (k, k), 0)

            # Downscale to cap the max side (memory/speed)
            if downscale_max_side and downscale_max_side > 0:
                h, w = bg.shape[:2]
                max_side = max(h, w)
                if max_side > downscale_max_side:
                    scale = downscale_max_side / float(max_side)
                    bg = cv2.resize(bg, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            bg_thumbs.append(bg)
            kept_ids.append(img_id)

        if len(bg_thumbs) == 0:
            _bprint(print_steps, "  No backgrounds in this batch; skipping.")
            continue

        _bprint(print_steps, f"Clustering {len(bg_thumbs)} backgrounds with {feature_extractor} features...")
        labels = _clusterByImgFeatures(
            bg_thumbs,
            feature_extractor=feature_extractor,
            full_display_ids=None,            # We want to display BG thumbs, not parent images
            data_folder=data_folder,
            print_steps=print_steps,
            cluster_params_dict=cluster_params_dict,
            show=True,
            show_save=show_save
        )

        # Visualize per-cluster grids of BACKGROUND thumbs
        uniq = sorted(set(labels))
        counts = Counter(labels)
        _bprint(print_steps, f"Found {len(uniq)} clusters: " + ", ".join(f"{c} (n={counts[c]})" for c in uniq))

        for c in uniq:
            imgs_c = [bg_thumbs[i] for i, lab in enumerate(labels) if lab == c]
            title = f"BG Cluster {c} (n={len(imgs_c)})"
            _showImages(True, imgs_c, sample_n=min(25, len(imgs_c)), maintitle=title, save_folder=save_folder)

        all_labels.extend(list(labels))
        all_kept_ids.extend(kept_ids)

    _bprint(print_steps, f"Done. Total clustered backgrounds: {len(all_kept_ids)}")
    return np.array(all_labels), all_kept_ids

clusterBackgrounds(data_folder="D:/bcc/chloros")