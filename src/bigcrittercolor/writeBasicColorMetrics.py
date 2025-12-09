import os
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from skimage.color import rgb2hsv, rgb2lab
import matplotlib.pyplot as plt

from bigcrittercolor.helpers import _readBCCImgs, _getBCCIDs, _showImages, makeCollage, _bprint
from bigcrittercolor.helpers.ids import _imgNameToID, _imgPathToName
from bigcrittercolor.helpers.image import _format

def writeBasicColorMetrics(img_ids=None, from_stage="segment", batch_size=1000,
                      get_color_metrics=True,
                      get_shape_texture_metrics=True,
                      threshold_metrics=[("hls",2,0.3,"below"),("rgb",0,0.6,"above")],
                      pattern_subfolder=None, show=False, print_steps=True, data_folder=''):

    """ Write color metrics like mean colors and thresholds directly obtained from filtered segments or color-clustered "patterns".

        Metrics appended to records are written to *data_folder*/records_with_metrics.csv.

        Args:
            from_stage (str): either "segment" or "pattern" (pattern referring to color clustered pattern)
            batch_size (int): get metrics in batches of this size - use to conserve memory when running tens or hundreds of thousands of images
            simple_metrics (bool): whether to get a suite of simple metrics like color channel means
            thresh_metrics (list):  a list of tuples defining the threshold metrics. The first element of each tuple is the colorspace (either "hls" or "rgb), the second element is the channel (0,1, or 2), the third element is the threshold value, and the fourth element is "above" or "below" to measure the percent area either above or below the threshold
    """

    _bprint(print_steps, "Starting writeSimpleColorMetrics...")
    # get all segment or pattern ids if None
    if img_ids is None:
        _bprint(print_steps, "No image ids specified, getting ids from all segments...")
        img_ids = _getBCCIDs(type=from_stage, data_folder=data_folder)

    # list to hold metrics
    all_metrics = []

    # do this in batches to avoid loading all images at once
    if batch_size is None:
        batch_size = len(img_ids)
    def process_batch(batch_img_ids, from_stage, show):
        # todo - make sure read works for patterns
        imgs = _readBCCImgs(type=from_stage,img_ids=batch_img_ids, data_folder=data_folder)

        if from_stage == "segment":
            # Start with a base DataFrame containing img_ids
            metrics = pd.DataFrame({'img_id': batch_img_ids})

            # Conditionally get and merge each type of metrics
            if get_color_metrics:
                simple_metrics = _getSimpleColorMetrics(imgs, batch_img_ids)
                metrics = pd.merge(metrics, simple_metrics, on='img_id', how='left')

            if threshold_metrics:  # Only if threshold_metrics is provided and not empty
                thresh_metrics = _getThresholdMetrics(imgs, batch_img_ids, thresholds=threshold_metrics, show=show)
                metrics = pd.merge(metrics, thresh_metrics, on='img_id', how='left')

            if get_shape_texture_metrics:
                shape_tex_metrics = _getShapeTextureMetrics(imgs, batch_img_ids, show=show)
                metrics = pd.merge(metrics, shape_tex_metrics, on='img_id', how='left')

        elif from_stage == "pattern":
            metrics = _getColorClusterMetrics(imgs, batch_img_ids)

        return metrics

    _bprint(print_steps, "Obtaining metrics in batches of size " + str(batch_size) + "...")
    # iterate over the image IDs in batches
    for i in range(0, len(img_ids), batch_size):
        batch_img_ids = img_ids[i:i + batch_size]
        batch_metrics = process_batch(batch_img_ids,from_stage,show)
        all_metrics.append(batch_metrics)
        print(str(i) + "/" + str(len(img_ids)))

    _bprint(print_steps, "Cleaning metric dataframe...")
    # concatenate all the batch metrics into a single DataFrame
    metrics = pd.concat(all_metrics, ignore_index=True)

    # while only doing 1 img per obs (temporary)
    metrics['obs_id'] = metrics['img_id'].str.replace('-1$', '', regex=True)

    metrics.to_csv(data_folder + "/metrics.csv",index=False)
    records = pd.read_csv(data_folder + "/records.csv")

    # while only doing 1 img per obs (temporary)
    records_with_metrics = pd.merge(metrics,records,on='obs_id')

    # Rename the column 'img_id_x' to 'img_id'
    records_with_metrics.rename(columns={'img_id_x': 'img_id'}, inplace=True)
    # Remove the column 'img_id_y'
    records_with_metrics.drop(columns=['img_id_y'], inplace=True)

    _bprint(print_steps, "Writing metric dataframe to " + data_folder + "/records_with_metrics.csv" + "...")
    records_with_metrics.to_csv(data_folder + "/records_with_metrics.csv",index=False)
    _bprint(print_steps, "Finished writeSimpleColorMetrics.")

# means, thresholds
def _getSimpleColorMetrics(imgs, img_ids):
    data = []

    for img, img_id in zip(imgs, img_ids):

        image = img
        # convert image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # mask to exclude black pixels
        mask = (image_rgb[:, :, 0] != 0) | (image_rgb[:, :, 1] != 0) | (image_rgb[:, :, 2] != 0)

        if not np.any(mask):
            # skip if no non-black pixels
            continue

        # Apply mask
        image_rgb = image_rgb[mask]

        # Compute mean RGB
        mean_red = np.mean(image_rgb[:, 0])
        mean_green = np.mean(image_rgb[:, 1])
        mean_blue = np.mean(image_rgb[:, 2])

        # Convert to HSV and compute mean hue and saturation
        hsv_image = rgb2hsv(image_rgb.reshape(-1, 1, 3) / 255.0)
        mean_hue = np.mean(hsv_image[:, :, 0])
        mean_saturation = np.mean(hsv_image[:, :, 1])
        mean_value = np.mean(hsv_image[:, :, 2])

        # Convert to CIELAB and compute mean lightness
        lab_image = rgb2lab(image_rgb.reshape(-1, 1, 3) / 255.0)
        mean_cielab_lightness = np.mean(lab_image[:, :, 0])

        # Append the metrics to the data list
        data.append({
            'img_id': img_id,
            'mean_red': mean_red,
            'mean_green': mean_green,
            'mean_blue': mean_blue,
            'mean_hue': mean_hue,
            'mean_saturation': mean_saturation,
            'mean_value': mean_value,
            'mean_cielab_lightness': mean_cielab_lightness
        })

    # Create a DataFrame from the data list
    df = pd.DataFrame(data)
    return df

import cv2
import numpy as np
import pandas as pd
from bigcrittercolor.helpers import _showImages, makeCollage


def _getThresholdMetrics(images, img_ids, thresholds=[("hls", 2, 0.3, "below")], show=False):
    """
    Enhanced thresholding:
      - Supports single thresholds (above/below) [backward compatible]
      - Supports range thresholds: (colorspace, channel, (low, high), 'between'|'outside')
      - Supports dual thresholds (AND):
          ('dual', condA, condB, optional_name)
      - Supports triple thresholds (AND):
          ('triple', condA, condB, condC, optional_name)
        where condA/condB/condC are single/range threshold specs as above.

    Hue handling:
      - If colorspace=='hls' and channel==0, any threshold/range values >1.0 are treated as degrees.
        Values <=180 are scaled by 180 (OpenCV hue), >180 by 360.
      - Range supports wrap-around when low > high (e.g., hue (350, 20) degrees).

    Returns: DataFrame with one row per img_id and one column per threshold spec (fraction of non-black pixels meeting the condition).
    If show=True, overlays satisfied pixels in red.
    """
    import math

    def _extract_channel(rgb):
        # returns dict of normalized channels in [0,1]
        hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS).astype(np.float32)
        # normalize to 0-1
        H = hls[:, :, 0] / 180.0  # OpenCV hue is 0-180
        L = hls[:, :, 1] / 255.0
        S = hls[:, :, 2] / 255.0
        R = rgb[:, :, 0].astype(np.float32) / 255.0
        G = rgb[:, :, 1].astype(np.float32) / 255.0
        B = rgb[:, :, 2].astype(np.float32) / 255.0
        return {
            ("hls", 0): H, ("hls", 1): L, ("hls", 2): S,
            ("rgb", 0): R, ("rgb", 1): G, ("rgb", 2): B
        }

    def _is_hue(colorspace, channel):
        return colorspace.lower() in ("hls", "hsv") and int(channel) == 0

    def _norm_scalar_thresh(colorspace, channel, t):
        if _is_hue(colorspace, channel) and t > 1:
            # Always scale by 360 for consistency when using degrees
            scale = 360.0
            return float(t) / scale
        return float(t)

    def _norm_range_thresh(colorspace, channel, rng):
        a, b = rng
        return (_norm_scalar_thresh(colorspace, channel, float(a)),
                _norm_scalar_thresh(colorspace, channel, float(b)))

    def _single_mask(rgb, nonblack_mask, spec):
        """
        spec forms:
          (colorspace, channel, thresh, 'above'|'below')
          (colorspace, channel, (low, high), 'between'|'outside')
        returns boolean mask (same shape as nonblack_mask)
        """
        colorspace, channel, thr, direction = spec
        colorspace = colorspace.lower()
        channel = int(channel)

        chan_maps = _extract_channel(rgb)
        if (colorspace, channel) not in chan_maps:
            raise ValueError(f"Unsupported colorspace/channel: {colorspace}, {channel}")
        vals = chan_maps[(colorspace, channel)]

        if isinstance(thr, (tuple, list)) and len(thr) == 2:
            low, high = _norm_range_thresh(colorspace, channel, thr)
            # handle wrap-around for cyclic hue ranges
            if _is_hue(colorspace, channel) and low > high:
                in_range = (vals >= low) | (vals <= high)
            else:
                in_range = (vals >= low) & (vals <= high)
            mask = in_range if direction == "between" else ~in_range
        else:
            t = _norm_scalar_thresh(colorspace, channel, float(thr))
            if direction == "below":
                mask = vals <= t
            elif direction == "above":
                mask = vals >= t
            else:
                raise ValueError(f"Unknown direction '{direction}' for single threshold")
        return mask & nonblack_mask

    def _parse_dual_name(a, b):
        def _name_of(spec):
            cs, ch, thr, direction = spec
            if isinstance(thr, (tuple, list)):
                lo, hi = thr
                return f"{cs}_ch{ch}_{direction}{lo}-{hi}"
            else:
                return f"{cs}_ch{ch}_{direction}{thr}"

        return f"dual_{_name_of(a)}__AND__{_name_of(b)}"

    def _parse_triple_name(a, b, c):
        def _name_of(spec):
            cs, ch, thr, direction = spec
            if isinstance(thr, (tuple, list)):
                lo, hi = thr
                return f"{cs}_ch{ch}_{direction}{lo}-{hi}"
            else:
                return f"{cs}_ch{ch}_{direction}{thr}"

        return f"triple_{_name_of(a)}__AND__{_name_of(b)}__AND__{_name_of(c)}"

    data = []
    imgs_to_show, titles = [], []

    for img, img_id in zip(images, img_ids):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        nonblack_mask = np.any(rgb != [0, 0, 0], axis=2)

        row = {"img_id": img_id}

        for spec in thresholds:
            # --- Triple threshold?
            if isinstance(spec, (tuple, list)) and len(spec) >= 4 and str(spec[0]).lower() == "triple":
                # ('triple', condA, condB, condC, optional_name)
                if len(spec) < 4:
                    raise ValueError("Triple threshold must be ('triple', condA, condB, condC, [name])")
                _, condA, condB, condC, *maybe_name = spec
                name = maybe_name[0] if maybe_name else _parse_triple_name(condA, condB, condC)

                maskA = _single_mask(rgb, nonblack_mask, condA)
                maskB = _single_mask(rgb, nonblack_mask, condB)
                maskC = _single_mask(rgb, nonblack_mask, condC)
                tmask = maskA & maskB & maskC  # AND all three conditions

                count = np.count_nonzero(nonblack_mask)
                frac = np.nan if count == 0 else (np.count_nonzero(tmask) / count)
                row[name] = frac

                if show:
                    marked = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
                    marked[tmask] = [0, 0, 255]
                    collage = makeCollage([img, marked], n_per_row=2)
                    imgs_to_show.append(collage)
                    titles.append(f"{name}: {0 if np.isnan(frac) else frac * 100:.1f}%")

            # --- Dual threshold?
            elif isinstance(spec, (tuple, list)) and len(spec) >= 3 and str(spec[0]).lower() == "dual":
                # ('dual', condA, condB, optional_name)
                if len(spec) < 3:
                    raise ValueError("Dual threshold must be ('dual', condA, condB, [name])")
                _, condA, condB, *maybe_name = spec
                name = maybe_name[0] if maybe_name else _parse_dual_name(condA, condB)

                maskA = _single_mask(rgb, nonblack_mask, condA)
                maskB = _single_mask(rgb, nonblack_mask, condB)
                tmask = maskA & maskB

                count = np.count_nonzero(nonblack_mask)
                frac = np.nan if count == 0 else (np.count_nonzero(tmask) / count)
                row[name] = frac

                if show:
                    marked = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
                    marked[tmask] = [0, 0, 255]
                    collage = makeCollage([img, marked], n_per_row=2)
                    imgs_to_show.append(collage)
                    titles.append(f"{name}: {0 if np.isnan(frac) else frac * 100:.1f}%")

            # --- Single/range threshold
            else:
                # Expect (colorspace, channel, thr or (low,high), direction)
                if not (isinstance(spec, (tuple, list)) and len(spec) == 4):
                    raise ValueError(
                        "Threshold spec must be "
                        "(colorspace, channel, thresh, 'above'|'below') "
                        "or (colorspace, channel, (low,high), 'between'|'outside') "
                        "or ('dual', condA, condB, [name]) "
                        "or ('triple', condA, condB, condC, [name])."
                    )
                colorspace, channel, thr, direction = spec
                tmask = _single_mask(rgb, nonblack_mask, (colorspace, channel, thr, direction))

                # build column name
                if isinstance(thr, (tuple, list)):
                    lo, hi = thr
                    key = f"{colorspace}_ch{channel}_{direction}{lo}-{hi}"
                else:
                    key = f"{colorspace}_ch{channel}_{direction}{thr}"

                count = np.count_nonzero(nonblack_mask)
                frac = np.nan if count == 0 else (np.count_nonzero(tmask) / count)
                row[key] = frac

                if show:
                    marked = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
                    marked[tmask] = [0, 0, 255]
                    collage = makeCollage([img, marked], n_per_row=2)
                    imgs_to_show.append(collage)
                    titles.append(f"{key}: {0 if np.isnan(frac) else frac * 100:.1f}%")

        data.append(row)

    # Fixed: Only show images if there are any to show, and limit both imgs and titles to sample_n
    if show and imgs_to_show:
        sample_n = 18  # Match the expected sample size
        # Limit both images and titles to the same sample size
        if len(imgs_to_show) > sample_n:
            imgs_to_show = imgs_to_show[:sample_n]
            titles = titles[:sample_n]
        _showImages(True, imgs_to_show, titles=titles, sample_n=len(imgs_to_show))

    return pd.DataFrame(data)



from skimage.measure import regionprops_table, label
from skimage.filters import gabor
from scipy.stats import kurtosis, entropy

def _getShapeTextureMetrics(imgs, img_ids, show=False):
    data = []

    for img, img_id in zip(imgs, img_ids):
        # Ensure image is grayscale for processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Binarize the image (assuming foreground is non-black)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        _showImages(show,images=[binary],maintitle="Binary")

        # Label the binary regions to extract connected components
        labeled_img = label(binary)
        props = regionprops_table(labeled_img, properties=[
            'major_axis_length', 'minor_axis_length', 'area', 'solidity', 'eccentricity'
        ])

        if len(props['area']) == 0:  # Skip if no objects detected
            continue

        # Extract shape metrics
        length = props['major_axis_length'][0]
        width = props['minor_axis_length'][0]
        area = props['area'][0]
        solidity = props['solidity'][0]
        eccentricity = props['eccentricity'][0]
        length_width_ratio = length / width if width > 0 else None

        # Circularity: 4π * Area / Perimeter²
        # Find contours in the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Check if any contours exist
        if len(contours) > 0:
            # Take the largest contour (or the first contour as default)
            largest_contour = max(contours, key=cv2.contourArea)

            # Compute the perimeter of the largest contour
            perimeter = cv2.arcLength(largest_contour, True)

            # Visualize the contour
            # Create a copy of the original binary image to draw on
            contour_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)  # Convert to color for visualization
            cv2.drawContours(contour_image, [largest_contour], -1, (0, 255, 0), 2)  # Green contour, thickness 2
            _showImages(show,images=[contour_image],maintitle="Contour")
        else:
            # No contours found; set perimeter to None or 0
            perimeter = 0
        #perimeter = cv2.arcLength(cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else None

        # Compute Hu Moments
        moments = cv2.moments(binary)
        hu_moments = cv2.HuMoments(moments).flatten()
        hu1, hu2, hu3, hu4 = hu_moments[:4]

        # Texture metrics
        non_zero_pixels = gray[gray > 0]
        entropy_value = entropy(np.histogram(non_zero_pixels, bins=256, density=True)[0])
        kurtosis_value = kurtosis(non_zero_pixels, axis=None)

        # Gabor filter responses
        gabor_horizontal = gabor(gray, frequency=0.2, theta=0)  # Horizontal: theta=0
        gabor_vertical = gabor(gray, frequency=0.2, theta=np.pi / 2)  # Vertical: theta=90°

        # Mean Gabor responses
        gabor_horizontal_mean = np.mean(np.abs(gabor_horizontal[0]))
        gabor_vertical_mean = np.mean(np.abs(gabor_vertical[0]))

        # Append to data
        data.append({
            'img_id': img_id,
            'length_pixels': length,
            'width_pixels': width,
            'area_pixels': area,
            'length_width_ratio': length_width_ratio,
            'circularity': circularity,
            'eccentricity_elongation': eccentricity,
            'solidity_nonconcavity': solidity,
            'hu1_xy_spread_size': hu1,
            'hu2_xy_asymmetry': hu2,
            'hu3_diagonal_asymmetry': hu3,
            'hu4_elongation': hu4,
            'entropy_irregularity': entropy_value,
            'kurtosis_dark_bright_tails': kurtosis_value,
            'gabor_horizontal': gabor_horizontal_mean,
            'gabor_vertical': gabor_vertical_mean
        })

    # Create a DataFrame
    df = pd.DataFrame(data)
    return df

def _getColorClusterMetrics(images, img_ids):
    """
    For a batch of pattern images, computes the proportion of each unique RGB color
    present in the batch (excluding background). Returns a wide DataFrame with:
      img_id, col_1_r, col_1_g, col_1_b, col_1_prop, col_2_r, ... etc.

    Fixes:
      - Convert BGR->RGB before analysis
      - Respect alpha if present (RGBA): keep only alpha>0
      - Deterministic color ordering (lexicographic by RGB)
    """
    import numpy as np
    import pandas as pd
    import cv2

    # --- Gather unique foreground colors across the batch (in RGB) ---
    all_colors = set()

    def _to_rgb_and_mask(img_bgr):
        # Handle BGR/RGBA safely
        if img_bgr.ndim != 3:
            return None, None  # skip not-3ch images
        h, w, c = img_bgr.shape
        if c == 4:
            # OpenCV doesn't carry alpha by default unless you loaded with IMREAD_UNCHANGED.
            # If c==4, assume BGRA.
            b, g, r, a = cv2.split(img_bgr)
            rgb = cv2.merge([r, g, b])
            fg_mask = (a > 0)
        elif c == 3:
            # BGR -> RGB
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            # treat pure black as background
            fg_mask = np.any(rgb != [0, 0, 0], axis=2)
        else:
            return None, None
        return rgb, fg_mask

    # First pass: collect batch palette
    for img in images:
        rgb, mask = _to_rgb_and_mask(img)
        if rgb is None or mask is None or not np.any(mask):
            continue
        # extract foreground pixels
        fg = rgb[mask]
        if fg.size == 0:
            continue
        # unique rows fast
        uniq = np.unique(fg, axis=0)
        for row in uniq:
            all_colors.add((int(row[0]), int(row[1]), int(row[2])))

    if not all_colors:
        # Nothing foreground-like; return just img_id column
        return pd.DataFrame({"img_id": img_ids})

    # Deterministic order for columns
    uniq_colors = np.array(sorted(all_colors, key=lambda t: (t[0], t[1], t[2])), dtype=np.int32)

    # --- Second pass: per-image proportions over the fixed palette ---
    data_rows = []
    for img, img_id in zip(images, img_ids):
        rgb, mask = _to_rgb_and_mask(img)
        row = {"img_id": img_id}
        # Initialize with zeros so column set is consistent even if some colors aren’t present
        for i, (r, g, b) in enumerate(uniq_colors, start=1):
            row[f"col_{i}_r"] = r
            row[f"col_{i}_g"] = g
            row[f"col_{i}_b"] = b
            row[f"col_{i}_prop"] = 0.0

        if rgb is None or mask is None or not np.any(mask):
            data_rows.append(row)
            continue

        fg = rgb[mask]
        total = fg.shape[0]
        if total == 0:
            data_rows.append(row)
            continue

        # Count occurrences of each color present in this image
        # Map color -> count using a structured view for speed
        fg_view = fg.view([('r', fg.dtype), ('g', fg.dtype), ('b', fg.dtype)]).reshape(-1)
        uniq_img, counts = np.unique(fg_view, return_counts=True)
        # Convert back to plain (r,g,b) tuples for matching
        uniq_img_rgb = np.column_stack([uniq_img['r'], uniq_img['g'], uniq_img['b']])

        # Build a dict for quick lookup
        # Note: using tuple keys to match uniq_colors tuples
        counts_dict = { (int(r), int(g), int(b)): int(c) for (r,g,b), c in zip(uniq_img_rgb, counts) }

        # Fill proportions for colors that appear
        for i, key in enumerate(map(tuple, uniq_colors), start=1):
            c = counts_dict.get(key, 0)
            if c:
                row[f"col_{i}_prop"] = c / float(total)

        data_rows.append(row)

    # Build columns in the same deterministic order
    cols = ["img_id"]
    for i in range(1, len(uniq_colors) + 1):
        cols += [f"col_{i}_r", f"col_{i}_g", f"col_{i}_b", f"col_{i}_prop"]

    df = pd.DataFrame(data_rows)[cols]
    return df