import os
import cv2
import numpy as np
import pandas as pd
import random
from skimage.color import rgb2lab
from bigcrittercolor.helpers import _readBCCImgs, _getBCCIDs, _showImages, makeCollage, _bprint

def writeMidtoneShare(
    img_ids=None,
    from_stage="segment",
    batch_size=1000,
    data_folder='',
    print_steps=True,
    # ---- visualization controls ----
    visualize_extremes=True,
    visualize_n=12,
    visualize_quantiles=False,
    quantile_probs=(0, 20, 40, 60, 80, 100),
    quantile_k=10,
    quantile_metric="illum_index",  # <---- NEW: choose "illum_index" or "midtone_share"
    # ---- outputs ----
    metrics_filename="midtone_share_metrics.csv",
    merged_filename="records_with_midtone_share.csv",
    # ---- metric tuning ----
    mid_band=(0.35, 0.65),
    robust_low=5,
    robust_high=95
):
    """
    Computes midtone-related metrics and allows visualization by selected metric.

    quantile_metric options:
      - "illum_index" (default): visualize illum balance
      - "midtone_share": visualize midtone proportion
    """

    _bprint(print_steps, "Starting writeMidtoneShare...")

    if img_ids is None:
        img_ids = _getBCCIDs(type=from_stage, data_folder=data_folder)
        _bprint(print_steps, f"No image ids specified, using all {len(img_ids)} {from_stage}s.")
    img_ids = random.sample(img_ids, 1000)
    if batch_size is None:
        batch_size = len(img_ids)

    mid_lo, mid_hi = mid_band

    def _to_rgb_and_fgmask(img_bgr_or_bgra):
        if img_bgr_or_bgra is None or img_bgr_or_bgra.ndim != 3:
            return None, None
        h, w, c = img_bgr_or_bgra.shape
        if c == 4:
            b, g, r, a = cv2.split(img_bgr_or_bgra)
            rgb = cv2.merge([r, g, b])
            fg = (a > 0)
        elif c == 3:
            rgb = cv2.cvtColor(img_bgr_or_bgra, cv2.COLOR_BGR2RGB)
            fg = np.any(rgb != [0, 0, 0], axis=2)
        else:
            return None, None
        return rgb.astype(np.uint8), fg

    def _metrics_batch(images, ids):
        rows, vis_records = [], []
        for img, img_id in zip(images, ids):
            row = {"img_id": img_id}
            rgb, fg = _to_rgb_and_fgmask(img)
            if rgb is None or fg is None or not np.any(fg):
                row.update({k: np.nan for k in ["n_nonblack_pixels","midtone_share","exposure_bias_median",
                                                "tail_imbalance","directional_midtone_score",
                                                "center10_90_bias","illum_index"]})
                rows.append(row)
                continue

            rgb_norm = rgb[fg].reshape(-1,1,3) / 255.0
            lab = rgb2lab(rgb_norm).reshape(-1,3)
            L = lab[:,0] / 100.0
            n = L.size
            row["n_nonblack_pixels"] = int(n)

            if n < 10:
                row.update({k: np.nan for k in ["midtone_share","exposure_bias_median",
                                                "tail_imbalance","directional_midtone_score",
                                                "center10_90_bias","illum_index"]})
                rows.append(row)
                continue

            L_low, L_high = np.percentile(L, robust_low), np.percentile(L, robust_high)
            if L_high <= L_low:
                row.update({k: np.nan for k in ["midtone_share","exposure_bias_median",
                                                "tail_imbalance","directional_midtone_score",
                                                "center10_90_bias","illum_index"]})
                rows.append(row)
                continue

            Ls = np.clip((L - L_low) / (L_high - L_low), 0.0, 1.0)
            mid_mask = (Ls >= mid_lo) & (Ls <= mid_hi)
            low_tail = (Ls < mid_lo)
            high_tail = (Ls > mid_hi)

            midtone_share = float(np.mean(mid_mask))
            exposure_bias_median = float(np.median(Ls) - 0.5)
            tail_imbalance = float(np.mean(high_tail) - np.mean(low_tail))
            directional_midtone_score = float((1.0 - midtone_share) * tail_imbalance)
            L10, L90 = np.percentile(L, 10), np.percentile(L, 90)
            if L90 > L10:
                Ls_10_90 = np.clip((L - L10) / (L90 - L10), 0.0, 1.0)
                center10_90_bias = float(np.median(Ls_10_90) - 0.5)
            else:
                center10_90_bias = np.nan

            severity = 1.0 - midtone_share
            direction_raw = 0.30*exposure_bias_median + 0.70*tail_imbalance
            if not np.isnan(center10_90_bias):
                direction_raw = 0.85*direction_raw + 0.15*center10_90_bias
            direction = float(np.clip(direction_raw, -1.0, 1.0))
            illum_index = float(severity * direction)

            row.update({
                "midtone_share": midtone_share,
                "exposure_bias_median": exposure_bias_median,
                "tail_imbalance": tail_imbalance,
                "directional_midtone_score": directional_midtone_score,
                "center10_90_bias": center10_90_bias,
                "illum_index": illum_index
            })
            rows.append(row)

            title = (f"id:{img_id}  mid:{midtone_share:.2f}  medBias:{exposure_bias_median:+.2f}  "
                     f"tails:{tail_imbalance:+.2f}  idx:{illum_index:+.2f}")
            vis_records.append((illum_index, midtone_share, img, title))
        return pd.DataFrame(rows), vis_records

    _bprint(print_steps, f"Computing midtone metrics in batches of {batch_size}...")
    all_metrics, all_vis = [], []

    for i in range(0, len(img_ids), batch_size):
        batch_ids = img_ids[i:i+batch_size]
        imgs = _readBCCImgs(type=from_stage, img_ids=batch_ids, data_folder=data_folder)
        dfb, visb = _metrics_batch(imgs, batch_ids)
        all_metrics.append(dfb)
        all_vis.extend(visb)
        print(f"{i}/{len(img_ids)}")

    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics['obs_id'] = metrics['img_id'].str.replace('-1$', '', regex=True)
    metrics.to_csv(os.path.join(data_folder, metrics_filename), index=False)

    # Merge with records
    rec_path = os.path.join(data_folder, "records.csv")
    records = pd.read_csv(rec_path)
    merged = pd.merge(metrics, records, on="obs_id", how="left")
    if 'img_id_x' in merged.columns: merged.rename(columns={'img_id_x':'img_id'}, inplace=True)
    if 'img_id_y' in merged.columns: merged.drop(columns=['img_id_y'], inplace=True)
    merged.to_csv(os.path.join(data_folder, merged_filename), index=False)

    _bprint(print_steps, f"Wrote metrics and merged CSVs.")

    if not all_vis:
        _bprint(print_steps, "No visualization: no valid images found.")
        return

    # ---- Extremes visualization (always based on illum_index) ----
    if visualize_extremes and visualize_n > 0:
        vis_sorted = sorted(all_vis, key=lambda t: t[0])
        k = max(1, visualize_n // 2)
        lows, highs = vis_sorted[:k], vis_sorted[-k:]
        imgs, titles = zip(*[(img, title) for _,_,img,title in lows+highs])
        _showImages(True, list(imgs), titles=list(titles))

    # ---- Quantile visualization ----
    if visualize_quantiles and quantile_k > 0:
        # Decide which metric to sort on
        metric_index = 1 if quantile_metric == "midtone_share" else 0
        vis_sorted = sorted(all_vis, key=lambda t: t[metric_index])
        metric_values = np.array([t[metric_index] for t in vis_sorted], dtype=np.float32)
        probs = [float(q*100.0) if 0 <= q <= 1 else float(q) for q in quantile_probs]
        probs = [min(100.0, max(0.0, p)) for p in probs]
        targets = np.percentile(metric_values, probs)

        used, imgs_to_show, titles = set(), [], []
        for q_pct, target in zip(probs, targets):
            idxs = []
            center = int(np.argmin(np.abs(metric_values - target)))
            left, right = center, center+1
            if center not in used:
                idxs.append(center); used.add(center)
            while len(idxs) < quantile_k and (left >= 0 or right < len(metric_values)):
                cand_left, cand_right = left-1, right
                dist_left = abs(metric_values[cand_left]-target) if cand_left >= 0 else np.inf
                dist_right = abs(metric_values[cand_right]-target) if cand_right < len(metric_values) else np.inf
                if dist_left <= dist_right:
                    if cand_left >= 0 and cand_left not in used:
                        idxs.append(cand_left); used.add(cand_left)
                    left = cand_left
                else:
                    if cand_right < len(metric_values) and cand_right not in used:
                        idxs.append(cand_right); used.add(cand_right)
                    right = cand_right + 1
                if (cand_left < 0 and cand_right >= len(metric_values)) or (dist_left==dist_right==np.inf):
                    break
            for rank, idx in enumerate(sorted(idxs)):
                illum, mid, img, base_title = vis_sorted[idx]
                metric_name = quantile_metric
                metric_val = metric_values[idx]
                header = f"[Q{q_pct:.0f}% target {metric_val:.2f}]"
                qtitle = f"{header} (#{rank+1}/{len(idxs)})  {base_title}"
                imgs_to_show.append(img)
                titles.append(qtitle)
        if imgs_to_show:
            _showImages(True, imgs_to_show, sample_n=None, num_cols=quantile_k)

    _bprint(print_steps, "Finished writeMidtoneShare.")

#from bigcrittercolor.helpers import _getBCCIDs
#ids = _getBCCIDs(type="image",sample_n=1000)
writeMidtoneShare(data_folder="D:/bcc/chloros", from_stage="image",visualize_extremes=False, visualize_quantiles=True,quantile_k=20,
                  quantile_probs=(0, 50, 100),quantile_metric="midtone_share")