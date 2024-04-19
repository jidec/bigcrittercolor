import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from bigcrittercolor.helpers.image import _format

def _scatterColors(color_values, input_colorspace, sample_n=500, pt_size=3, cluster_labels=None):
    """
    Plot colors in a specified input color space (HLS or CIELAB or RGB), coloring each point by its actual color
    (converted to RGB if necessary) and changing the marker shape based on cluster ID.
    """

    if sample_n is not None and color_values.shape[0] > sample_n:
        indices = random.sample(range(len(color_values)), sample_n)
        color_values = color_values[indices]
        cluster_labels = [cluster_labels[i] for i in indices]

    # Define a set of markers to cycle through
    markers = ['o', 's', '^', 'P', '*', 'X', 'D']

    # Reshape the color_values for OpenCV and convert to RGB if necessary
    reshaped_colors = color_values.reshape((-1, 1, 3))
    if input_colorspace.lower() == 'hls':
        point_colors = cv2.cvtColor(np.uint8(reshaped_colors), cv2.COLOR_HLS2RGB)
    elif input_colorspace.lower() == 'cielab':
        point_colors = cv2.cvtColor(np.uint8(reshaped_colors), cv2.COLOR_Lab2BGR) #RGB
        #print("COLORS IN SCATTER")
        #print(color_values)
        #print(point_colors)
    else:
        point_colors = color_values

    #point_colors = _format()
    # Flatten the point_colors back to original shape
    point_colors = point_colors.reshape((-1, 3))

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(color_values)):
        xs, ys, zs = color_values[i, 0], color_values[i, 1], color_values[i, 2] #0,1,1
        color = point_colors[i] / 255.0
        marker = markers[cluster_labels[i] % len(markers)] if cluster_labels is not None else 'o'

        ax.scatter(xs, ys, zs, c=[color], marker=marker, s=pt_size)

    channel_names = {"rgb": ["Red", "Green", "Blue"],
                     "cielab": ["Lightness", "Red-To-Green", "Blue-To-Yellow"],
                     "hls": ["Hue", "Lightness", "Saturation"]}

    ax.set_xlabel(channel_names[input_colorspace][0])
    ax.set_ylabel(channel_names[input_colorspace][1])
    ax.set_zlabel(channel_names[input_colorspace][2])

    plt.show()

def _scatterColors_old(rgb_values, colorspace=None, sample_n= 250, pt_size=5, cluster_labels=None):
    """
    Plot RGB colors in a specified color space (HLS or CIELAB),
    coloring each point by its actual RGB color and changing the marker shape based on cluster ID.

    :param rgb_values: numpy array of RGB colors (shape Nx3).
    :param colorspace: Optional; 'hls' or 'cielab'. If None, uses RGB.
    :param cluster_labels: Optional; List of cluster IDs for each color.
    """

    if sample_n is not None:
        indices = random.sample(range(len(rgb_values)), sample_n)
        rgb_values = rgb_values[indices]
        cluster_labels = [cluster_labels[i] for i in indices]

    # Define a set of markers to cycle through
    markers = ['o', 's', '^', 'P', '*', 'X', 'D']

    # Convert RGB to the specified color space
    if colorspace is not None:
        if colorspace.lower() == 'hls':
            colors = cv2.cvtColor(np.uint8([rgb_values]), cv2.COLOR_RGB2HLS)[0]
        elif colorspace.lower() == 'cielab':
            colors = cv2.cvtColor(np.uint8([rgb_values]), cv2.COLOR_RGB2Lab)[0]
        else:
            raise ValueError("Colorspace not supported. Choose 'hls', 'cielab', or None.")
    else:
        colors = rgb_values
        #colors = cv2.cvtColor(np.uint8([rgb_values]), cv2.COLOR_RGB2BGR)[0]

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(rgb_values)):
        xs, ys, zs = colors[i, 0], colors[i, 1], colors[i, 2]
        color = np.uint8(rgb_values[i]) / 255.0
        marker = markers[cluster_labels[i] % len(markers)] if cluster_labels is not None else 'o'

        ax.scatter(xs, ys, zs, c=[color], marker=marker,s=pt_size)

    channel_names = ["Red","Green","Blue"]
    if colorspace == "cielab":
        channel_names = ["Lightness","Red-To-Green","Blue-To-Yellow"]
    if colorspace == "hls":
        channel_names = ["Hue","Lightness","Saturation"]

    ax.set_xlabel(channel_names[0])
    ax.set_ylabel(channel_names[1])
    ax.set_zlabel(channel_names[2])
    plt.show()

# Example usage:
#rgb_values = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
#cluster_labels = [1, 1, 2]
#_scatterColors(rgb_values, 'hls', cluster_labels)