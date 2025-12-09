import os
import re
import cv2
import numpy as np
import pandas as pd
from skimage.color import rgb2hsv
from sklearn.neighbors import KDTree

from bigcrittercolor.helpers import _bprint
from bigcrittercolor.helpers import _getBCCIDs

def writeColorClusterMetrics(img_ids=None,
                             pattern_subfolder=None,
                             batch_size=1000,
                             data_folder='',
                             print_steps=True,
                             show=False,  # reserved for parity; not used here
                             merge_with_records=True,
                             output_metrics_name="color_cluster_metrics.csv",
                             output_palette_name="color_cluster_palette.csv",
                             tolerance=0):
    """
    Compute percent coverage of each *final pattern color* per image, with a
    global, consistent cluster palette across all images.

    Saves:
      - {data_folder}/{output_metrics_name}
      - {data_folder}/{output_palette_name}

    Args:
        img_ids (list|None): If None, auto-discovers from /patterns[/subfolder]/*_pattern.png
        pattern_subfolder (str|None): If patterns were written to a subfolder.
        batch_size (int): Process images in batches to reduce memory.
        data_folder (str): Project root that contains /patterns and /records.csv
        merge_with_records (bool): Merge per-image metrics with records.csv on obs_id
        tolerance (int): If >0, allows small RGB diffs (0–255 space) when mapping colors to the palette.
                         0 = exact match (fastest); >0 uses KDTree for nearest match within radius.
    """

    def _pattern_path_for(img_id):
        base = os.path.join(data_folder, "patterns")
        if pattern_subfolder:
            base = os.path.join(base, pattern_subfolder)
        return os.path.join(base, f"{img_id}_pattern.png")

    # -----------------------------
    # Discover image IDs if needed
    # -----------------------------
    if img_ids is None:
        _bprint(print_steps, "Discovering pattern image IDs...")
        base = os.path.join(data_folder, "patterns")
        if pattern_subfolder:
            base = os.path.join(base, pattern_subfolder)
        if not os.path.isdir(base):
            raise FileNotFoundError(f"Patterns folder not found: {base}")

        ids = []
        for fn in os.listdir(base):
            if fn.endswith("_pattern.png"):
                # strip '_pattern.png' to recover img_id
                img_id = re.sub(r"_pattern\.png$", "", fn)
                ids.append(img_id)
        img_ids = sorted(ids)

    if len(img_ids) == 0:
        _bprint(True, "No pattern images found. Exiting.")
        return

    _bprint(print_steps, f"Found {len(img_ids)} pattern images.")

    # -----------------------------
    # First pass: build global palette (unique non-background RGBs)
    # -----------------------------
    _bprint(print_steps, "Building global palette of unique colors...")
    palette_set = set()

    def _collect_palette_for_batch(batch_ids):
        local = set()
        for img_id in batch_ids:
            path = _pattern_path_for(img_id)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # keep alpha if present
            if img is None:
                continue

            # Separate channels
            if img.ndim == 3 and img.shape[2] == 4:
                bgr = img[:, :, :3]
                alpha = img[:, :, 3]
                fg_mask = alpha > 0
            else:
                bgr = img
                # Treat pure black as background
                fg_mask = np.any(bgr != [0, 0, 0], axis=2)

            # Extract RGB triplets of foreground pixels
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            fg_rgb = rgb[fg_mask]
            if fg_rgb.size == 0:
                continue

            uniq = np.unique(fg_rgb.reshape(-1, 3), axis=0)
            for t in uniq:
                local.add((int(t[0]), int(t[1]), int(t[2])))
        return local

    if batch_size is None:
        batch_size = len(img_ids)

    for i in range(0, len(img_ids), batch_size):
        batch = img_ids[i:i+batch_size]
        palette_set |= _collect_palette_for_batch(batch)
        _bprint(print_steps, f"Palette scan: {min(i+batch_size, len(img_ids))}/{len(img_ids)}")

    if len(palette_set) == 0:
        _bprint(True, "No non-background colors found in patterns. Exiting.")
        return

    # Convert to array and sort palette by HSV for consistent ordering (and nicer reading)
    palette_rgb = np.array(sorted(list(palette_set)), dtype=np.uint8)  # shape (K, 3)
    hsv = (rgb2hsv(palette_rgb[np.newaxis, :, :] / 255.0)[0])  # in [0,1]
    # lexsort by H, then S, then V
    order = np.lexsort((hsv[:, 2], hsv[:, 1], hsv[:, 0]))
    palette_rgb = palette_rgb[order]
    hsv = hsv[order]
    K = palette_rgb.shape[0]
    _bprint(print_steps, f"Global palette size: K={K}")

    # Optional tolerance: build KDTree for palette lookups (in RGB)
    tree = KDTree(palette_rgb.astype(np.float32)) if tolerance > 0 else None

    # -----------------------------
    # Second pass: per-image coverage
    # -----------------------------
    _bprint(print_steps, "Computing per-image coverage for each palette color...")

    # Prebuild column names
    cluster_cols = [f"cluster_{i+1:03d}_prop" for i in range(K)]
    # Also save palette table with RGB + HSV for reference
    palette_df = pd.DataFrame({
        "cluster_id": [f"cluster_{i+1:03d}" for i in range(K)],
        "r": palette_rgb[:, 0].astype(int),
        "g": palette_rgb[:, 1].astype(int),
        "b": palette_rgb[:, 2].astype(int),
        "h": hsv[:, 0],
        "s": hsv[:, 1],
        "v": hsv[:, 2],
    })

    rows = []

    def _process_batch(batch_ids):
        batch_rows = []
        for img_id in batch_ids:
            path = _pattern_path_for(img_id)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            if img.ndim == 3 and img.shape[2] == 4:
                bgr = img[:, :, :3]
                alpha = img[:, :, 3]
                fg_mask = alpha > 0
            else:
                bgr = img
                fg_mask = np.any(bgr != [0, 0, 0], axis=2)

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            fg_rgb = rgb[fg_mask]
            total = fg_rgb.shape[0]
            if total == 0:
                # all background; zero coverage
                row = {"img_id": img_id}
                row.update({c: 0.0 for c in cluster_cols})
                batch_rows.append(row)
                continue

            if tolerance == 0:
                # Fast exact counting using numpy grouping
                uniq, counts = np.unique(fg_rgb.reshape(-1, 3), axis=0, return_counts=True)
                # Map each unique color to palette index via dict
                # Build dict once per batch (cost negligible because K is small)
                pal_dict = {tuple(map(int, c)): i for i, c in enumerate(palette_rgb)}
                acc = np.zeros(K, dtype=np.int64)
                for col, cnt in zip(uniq, counts):
                    idx = pal_dict.get((int(col[0]), int(col[1]), int(col[2])), None)
                    if idx is not None:
                        acc[idx] += cnt
                    # else: if a color isn't in palette (shouldn't happen), ignore
            else:
                # Nearest within tolerance
                # Query palette for all pixels in one shot
                dist, idx = tree.query(fg_rgb.astype(np.float32), k=1, return_distance=True)
                idx = idx[:, 0]
                dist = dist[:, 0]
                # Keep only assignments within tolerance (RGB euclidean)
                valid = dist <= tolerance
                acc = np.zeros(K, dtype=np.int64)
                if np.any(valid):
                    # bincount counts per cluster index
                    counts = np.bincount(idx[valid], minlength=K)
                    acc[:len(counts)] += counts

            props = (acc / total).astype(float)
            row = {"img_id": img_id}
            for j, colname in enumerate(cluster_cols):
                row[colname] = props[j]
            batch_rows.append(row)

        return batch_rows

    for i in range(0, len(img_ids), batch_size):
        batch = img_ids[i:i+batch_size]
        rows.extend(_process_batch(batch))
        _bprint(print_steps, f"Coverage: {min(i+batch_size, len(img_ids))}/{len(img_ids)}")

    metrics_df = pd.DataFrame(rows, columns=["img_id"] + cluster_cols)

    # Add obs_id for merging convenience (mirrors your other writer)
    metrics_df["obs_id"] = metrics_df["img_id"].str.replace(r"-1$", "", regex=True)

    # -----------------------------
    # Save outputs
    # -----------------------------
    metrics_path = os.path.join(data_folder, output_metrics_name)
    palette_path = os.path.join(data_folder, output_palette_name)

    _bprint(print_steps, f"Writing metrics -> {metrics_path}")
    metrics_df.to_csv(metrics_path, index=False)

    _bprint(print_steps, f"Writing palette -> {palette_path}")
    palette_df.to_csv(palette_path, index=False)

    if merge_with_records:
        records_path = os.path.join(data_folder, "records.csv")
        if os.path.exists(records_path):
            records = pd.read_csv(records_path)
            merged = pd.merge(metrics_df, records, on="obs_id", how="left")
            out_merged = os.path.join(data_folder, "records_with_color_clusters.csv")
            _bprint(print_steps, f"Merging with records -> {out_merged}")
            merged.to_csv(out_merged, index=False)
        else:
            _bprint(print_steps, f"No records.csv at {records_path}; skipping merge.")

    _bprint(print_steps, "Done with writeColorClusterMetrics.")

writeColorClusterMetrics(data_folder="D:/bcc/chloros")