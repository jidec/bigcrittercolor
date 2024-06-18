import os
import numpy as np
from PIL import Image
import pandas as pd

def writeMetricsFromPatterns(data_folder, pattern_subfolder=None):
    img_dir = data_folder + "/patterns"
    if pattern_subfolder is not None:
        img_dir = img_dir + "/" + pattern_subfolder
    start_index = 1
    # Get paths
    paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png')]
    paths = paths[start_index - 1:]

    # Loop through first to find all unique colors
    all_colors = set()
    for i, path in enumerate(paths, start=start_index):
        if i % 100 == 0:
            print(i)
        img = Image.open(path)
        arr = np.array(img)
        # Remove black background pixels and flatten to list of colors
        arr = arr[(arr != [0, 0, 0, 0]).all(axis=2)]
        for color in np.unique(arr, axis=0):
            all_colors.add(tuple(color))

    uniq_colors = np.array(list(all_colors))
    # Loop through again to get props and build df
    data = []
    for i, path in enumerate(paths, start=start_index):
        if i % 100 == 0:
            print(i)
        img = Image.open(path)
        arr = np.array(img)
        arr = arr[(arr != [0, 0, 0, 0]).all(axis=2)]
        name = os.path.basename(path).split('_')[0]

        # Calculate mean color
        mean_rgb = arr.mean(axis=0)[:3]
        row = [name] + list(mean_rgb)

        # Calculate proportion of pixels for each unique color
        for color in uniq_colors:
            col_pix = np.sum((arr[:, :3] == color[:3]).all(axis=1))
            total_pix = arr.shape[0]
            prop = col_pix / total_pix if total_pix else 0
            row.extend(list(color[:3]) + [prop])

        data.append(row)

    # Creating DataFrame
    columns = ['img_id', 'mean_r', 'mean_g', 'mean_b']
    for i, color in enumerate(uniq_colors, start=1):
        columns.extend([f'col_{i}_r', f'col_{i}_g', f'col_{i}_b', f'col_{i}_prop'])

    df = pd.DataFrame(data, columns=columns)
    for i, color in enumerate(uniq_colors):
        colname_r = 'col_' + str(i + 1) + '_r'
        colname_g = 'col_' + str(i + 1) + '_g'
        colname_b = 'col_' + str(i + 1) + '_b'
        df[colname_r] = color[0]
        df[colname_g] = color[1]
        df[colname_b] = color[2]

    df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric)
    df['mean_lightness'] = df[['mean_r', 'mean_g', 'mean_b']].mean(axis=1)

    # TEMP while only doing 1 img per obs
    df['obs_id'] = df['img_id'].str.replace('-1$', '', regex=True)

    df.to_csv(data_folder + "/metrics.csv",index=False)
    records = pd.read_csv(data_folder + "/records.csv")

    # TEMP while only doing 1 img per obs
    records_with_metrics = pd.merge(df,records,on='obs_id')

    records_with_metrics.to_csv(data_folder + "/records_with_metrics.csv",index=False)

#writeMetricsFromPatterns("D:/bcc/ringtails")