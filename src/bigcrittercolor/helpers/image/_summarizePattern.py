import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def _summarizePattern(img_path):
    """
    Plot the different RGB colors in the image in a column of color squares using matplotlib,
    displaying the percentage of the image taken up by each color and the RGB values.
    Pixels with RGB values summing to less than 10 are excluded.

    :param img_path: Path to the image file.
    """
    # Read the image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Reshape the image to a 2D array of RGB values and filter out very dark pixels
    reshaped_img = img_rgb.reshape(-1, 3)
    filtered_pixels = np.array([pixel for pixel in reshaped_img if np.sum(pixel) >= 10])

    # Count the occurrences of each color
    color_counts = Counter(map(tuple, filtered_pixels))

    # Calculate total count of non-very-dark pixels
    total_count = sum(color_counts.values())

    # Sort colors by frequency
    sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)

    # Plotting
    plt.figure(figsize=(6, len(sorted_colors)))
    for i, (color, count) in enumerate(sorted_colors):
        plt.fill_betweenx([i - 0.5, i + 0.5], 0, 1, color=np.array(color) / 255.0)
        percentage = (count / total_count) * 100
        text_color = 'white' if sum(color) < 384 else 'black'
        plt.text(0.3, i, f"{color}", va='center', ha='left', color=text_color)
        plt.text(0.7, i, f"{percentage:.2f}%", va='center', ha='right', color=text_color)

    plt.yticks([])
    plt.xticks([])
    plt.title("Color Distribution in Image")
    plt.show()

# Example usage:
#_summarizePattern('D:/bcc/beetles/patterns/INATRANDOM-170943445_pattern.png')
#_summarizePattern('D:/bcc/beetles/patterns/INATRANDOM-5226102_pattern.png')
#_summarizePattern('D:/bcc/beetles/patterns/INATRANDOM-6814284_pattern.png')
#_summarizePattern('D:/bcc/beetles/patterns/INATRANDOM-27487856_pattern.png')