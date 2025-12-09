import os
import cv2
import numpy as np
import pandas as pd

from bigcrittercolor.helpers import _readBCCImgs, _getBCCIDs, _showImages, makeCollage, _bprint

def writeIllumClipping(
    img_ids=None,
    from_stage="segment",
    batch_size=1000,
    hi_thresh=0.98,           # >= this is considered high clip; accepts 0..1 or 0..255
    lo_thresh=0.02,           # <= this is considered low clip; accepts 0..1 or 0..255
    include_per_channel=True, # also report per-channel clip fractions
    show=False,               # visualize masks on a sample
    print_steps=True,
    data_folder='',
    metrics_filename="illum_metrics.csv",
    merged_filename="records_with_illum.csv"
):
    """
    Compute illumination clipping metrics per image, focusing on *synchronous channel clipping*
    (i.e., all RGB channels clipped together), after excluding black background pixels.

    Outputs:
      - {data_folder}/{metrics_filename}
      - {data_folder}/{merged_filename}  (merged with records.csv via obs_id)

    Metrics per image:
      - clip_high_all_frac        : fraction of non-black pixels with R>=hi & G>=hi & B>=hi
      - clip_high_any_frac        : fraction with any channel >=hi
      - clip_high_sync_ratio      : clip_high_all_frac / clip_high_any_frac
      - clip_low_all_frac         : fraction with R<=lo & G<=lo & B<=lo  (non-black pixels only)
      - clip_low_any_frac         : fraction with any channel <=lo       (non-black pixels only)
      - clip_low_sync_ratio       : clip_low_all_frac / clip_low_any_frac
      - [optional per-channel]: clip_high_R_frac, clip_high_G_frac, clip_high_B_frac,
                                clip_low_R_frac,  clip_low_G_frac,  clip_low_B_frac
      - n_nonblack_pixels         : denominator used for fractions

    Notes:
      - Thresholds accept normalized (0..1) or 8-bit (0..255) values.
      - Non-black mask excludes exact [0,0,0] RGB pixels and, for 4-channel images, alpha==0.
    """

    def _to_255(v):
        # Accept 0..1 or 0..255
        return int(round(v * 255)) if 0 <= v <= 1 else int(round(v))

    hi = _to_255(hi_thresh)
    lo = _to_255(lo_thresh)
    hi = max(0, min(255, hi))
    lo = max(0, min(255, lo))

    _bprint(print_steps, "Starting writeIllumClipping...")

    # get all segment or pattern ids if None
    if img_ids is None:
        _bprint(print_steps, f"No image ids specified, getting ids from all {from_stage}s...")
        img_ids = _getBCCIDs(type=from_stage, data_folder=data_folder)

    # local helper to handle BGR/BGRA → RGB + non-black mask
    def _rgb_and_mask(img_bgr_or_bgra):
        if img_bgr_or_bgra.ndim != 3:
            return None, None
        h, w, c = img_bgr_or_bgra.shape
        if c == 4:
            b, g, r, a = cv2.split(img_bgr_or_bgra)
            rgb = cv2.merge([r, g, b])
            nonblack = (a > 0)  # trust alpha for foreground
        elif c == 3:
            rgb = cv2.cvtColor(img_bgr_or_bgra, cv2.COLOR_BGR2RGB)
            nonblack = np.any(rgb != [0, 0, 0], axis=2)  # drop pure black background
        else:
            return None, None
        return rgb, nonblack

    def _illumClippingMetrics(images, ids, show=False):
        rows = []
        viz_imgs, viz_titles = [], []

        for img, img_id in zip(images, ids):
            rgb, nonblack = _rgb_and_mask(img)
            row = {"img_id": img_id}

            if rgb is None or nonblack is None:
                # still return row with NaNs for robustness
                row.update({
                    "clip_high_all_frac": np.nan, "clip_high_any_frac": np.nan, "clip_high_sync_ratio": np.nan,
                    "clip_low_all_frac":  np.nan, "clip_low_any_frac":  np.nan, "clip_low_sync_ratio":  np.nan,
                    "n_nonblack_pixels":  0
                })
                if include_per_channel:
                    row.update({
                        "clip_high_R_frac": np.nan, "clip_high_G_frac": np.nan, "clip_high_B_frac": np.nan,
                        "clip_low_R_frac":  np.nan, "clip_low_G_frac":  np.nan, "clip_low_B_frac":  np.nan
                    })
                rows.append(row)
                continue

            R = rgb[:, :, 0]
            G = rgb[:, :, 1]
            B = rgb[:, :, 2]

            # apply non-black mask
            Rm = R[nonblack]
            Gm = G[nonblack]
            Bm = B[nonblack]

            n = Rm.size
            row["n_nonblack_pixels"] = int(n)

            if n == 0:
                row.update({
                    "clip_high_all_frac": np.nan, "clip_high_any_frac": np.nan, "clip_high_sync_ratio": np.nan,
                    "clip_low_all_frac":  np.nan, "clip_low_any_frac":  np.nan, "clip_low_sync_ratio":  np.nan
                })
                if include_per_channel:
                    row.update({
                        "clip_high_R_frac": np.nan, "clip_high_G_frac": np.nan, "clip_high_B_frac": np.nan,
                        "clip_low_R_frac":  np.nan, "clip_low_G_frac":  np.nan, "clip_low_B_frac":  np.nan
                    })
                rows.append(row)
                continue

            # per-channel clip masks (1D after masking)
            r_hi = (Rm >= hi); g_hi = (Gm >= hi); b_hi = (Bm >= hi)
            r_lo = (Rm <= lo); g_lo = (Gm <= lo); b_lo = (Bm <= lo)

            # any vs all (synchronous) clipping
            hi_any = (r_hi | g_hi | b_hi)
            hi_all = (r_hi & g_hi & b_hi)

            lo_any = (r_lo | g_lo | b_lo)
            lo_all = (r_lo & g_lo & b_lo)

            # fractions
            hi_any_frac = float(np.count_nonzero(hi_any)) / n
            hi_all_frac = float(np.count_nonzero(hi_all)) / n
            lo_any_frac = float(np.count_nonzero(lo_any)) / n
            lo_all_frac = float(np.count_nonzero(lo_all)) / n

            row["clip_high_any_frac"] = hi_any_frac
            row["clip_high_all_frac"] = hi_all_frac
            row["clip_high_sync_ratio"] = (hi_all_frac / hi_any_frac) if hi_any_frac > 0 else np.nan

            row["clip_low_any_frac"] = lo_any_frac
            row["clip_low_all_frac"] = lo_all_frac
            row["clip_low_sync_ratio"] = (lo_all_frac / lo_any_frac) if lo_any_frac > 0 else np.nan

            if include_per_channel:
                row["clip_high_R_frac"] = float(np.count_nonzero(r_hi)) / n
                row["clip_high_G_frac"] = float(np.count_nonzero(g_hi)) / n
                row["clip_high_B_frac"] = float(np.count_nonzero(b_hi)) / n
                row["clip_low_R_frac"]  = float(np.count_nonzero(r_lo)) / n
                row["clip_low_G_frac"]  = float(np.count_nonzero(g_lo)) / n
                row["clip_low_B_frac"]  = float(np.count_nonzero(b_lo)) / n

            # optional visualization (mark synchronous clips)
            if show:
                # build full-image boolean masks for synchronous regions
                hi_all_full = np.zeros(nonblack.shape, dtype=bool)
                lo_all_full = np.zeros(nonblack.shape, dtype=bool)
                hi_any_full = np.zeros(nonblack.shape, dtype=bool)
                lo_any_full = np.zeros(nonblack.shape, dtype=bool)

                # place masked values back into full-size arrays
                hi_all_full[nonblack] = hi_all
                lo_all_full[nonblack] = lo_all
                hi_any_full[nonblack] = hi_any
                lo_any_full[nonblack] = lo_any

                # overlay on original BGR for display
                marked = img.copy()

                # color coding:
                #   synchronous highlights (all-high): RED
                #   synchronous shadows   (all-low) : BLUE
                #   any-only highlights (but not all): ORANGE
                #   any-only shadows   (but not all): CYAN
                any_only_hi = hi_any_full & (~hi_all_full)
                any_only_lo = lo_any_full & (~lo_all_full)

                marked[hi_all_full] = [0, 0, 255]      # red
                marked[lo_all_full] = [255, 0, 0]      # blue
                marked[any_only_hi] = [0, 165, 255]    # orange
                marked[any_only_lo] = [255, 255, 0]    # cyan

                collage = makeCollage([img, marked], n_per_row=2)
                title = (
                    f"hi_all:{hi_all_frac:.3f}  hi_any:{hi_any_frac:.3f}  "
                    f"lo_all:{lo_all_frac:.3f}  lo_any:{lo_any_frac:.3f}"
                )
                viz_imgs.append(collage)
                viz_titles.append(title)

            rows.append(row)

        if show and viz_imgs:
            # Sample BOTH images and titles so lengths match _showImages' expectation
            sample_n = 18
            if len(viz_imgs) > sample_n:
                # deterministic spread (or use random.sample for stochastic)
                idxs = np.linspace(0, len(viz_imgs) - 1, sample_n, dtype=int)
                viz_imgs_sampled = [viz_imgs[i] for i in idxs]
                viz_titles_sampled = [viz_titles[i] for i in idxs]
            else:
                viz_imgs_sampled = viz_imgs
                viz_titles_sampled = viz_titles

            _showImages(True, viz_imgs_sampled, titles=viz_titles_sampled, sample_n=None)

        return pd.DataFrame(rows)

    # ---- batching ----
    if batch_size is None:
        batch_size = len(img_ids)

    _bprint(print_steps, f"Computing clipping metrics in batches of {batch_size}...")

    all_metrics = []
    for i in range(0, len(img_ids), batch_size):
        batch_img_ids = img_ids[i:i + batch_size]
        imgs = _readBCCImgs(type=from_stage, img_ids=batch_img_ids, data_folder=data_folder)
        batch_df = _illumClippingMetrics(imgs, batch_img_ids, show=show)
        all_metrics.append(batch_df)
        print(f"{i}/{len(img_ids)}")

    _bprint(print_steps, "Concatenating and cleaning metric dataframe...")
    metrics = pd.concat(all_metrics, ignore_index=True)

    # while only doing 1 img per obs (temporary)
    metrics['obs_id'] = metrics['img_id'].str.replace('-1$', '', regex=True)

    # write metrics csv (separate filename to avoid overwriting other metrics)
    metrics_path = os.path.join(data_folder, metrics_filename)
    metrics.to_csv(metrics_path, index=False)

    # merge with records.csv
    records_path = os.path.join(data_folder, "records.csv")
    if not os.path.exists(records_path):
        raise FileNotFoundError(f"records.csv not found at {records_path}")

    records = pd.read_csv(records_path)

    # while only doing 1 img per obs (temporary)
    records_with_metrics = pd.merge(metrics, records, on='obs_id', how='left')

    # align img_id column names like your original flow
    if 'img_id_x' in records_with_metrics.columns:
        records_with_metrics.rename(columns={'img_id_x': 'img_id'}, inplace=True)
    if 'img_id_y' in records_with_metrics.columns:
        records_with_metrics.drop(columns=['img_id_y'], inplace=True)

    merged_path = os.path.join(data_folder, merged_filename)
    _bprint(print_steps, f"Writing merged dataframe to {merged_path} ...")
    records_with_metrics.to_csv(merged_path, index=False)

    _bprint(print_steps, "Finished writeIllumClipping.")

writeIllumClipping(data_folder="D:/bcc/chloros",show=True)