import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb, rgb2hsv

# Generate colors in CIELAB
n = 27  # Number of colors
L = np.linspace(20, 80, 3)  # Spanning lightness values from dark to light
a = np.linspace(-80, 80, 3)  # Spanning from green to red
b = np.linspace(-80, 80, 3)  # Spanning from blue to yellow

# Create a meshgrid of a, b, L values
A, B, L = np.meshgrid(a, b, L)
lab_colors = np.dstack((L.flatten(), A.flatten(), B.flatten()))

# Convert LAB to RGB for visualization
rgb_colors = lab2rgb(lab_colors.reshape((n, 1, 3))).reshape((n, 3))

# Convert RGB to HSV and sort by hue
hsv_colors = rgb2hsv(rgb_colors)
sorted_indices = np.argsort(hsv_colors[:, 0])
sorted_rgb_colors = rgb_colors[sorted_indices]

# Custom names provided
custom_color_names = [
    "Dark Red", "Red", "Orange", "Brown", "Light Brown", "Yellow",
    "Bright Green", "Green", "Dark Green", "Teal", "Sky Blue", "Light Blue",
    "Blue", "Indigo", "Dark Blue", "Violet", "Purple", "Magenta",
    "Pink", "Light Pink", "White", "Light Gray", "Gray",
    "Dark Gray", "Black", "Charcoal", "Deep Purple"
]

custom_color_names.reverse()

# Create a vertical plot for the sorted 27 colors with custom names
fig, ax = plt.subplots(figsize=(6, 12))
ax.imshow(sorted_rgb_colors[:, np.newaxis, :], aspect='auto', extent=[-10, 1, 0, 27])
ax.axis('off')  # Don't show axis

# Add custom color names vertically
for i, name in enumerate(custom_color_names):
    ax.text(1.1, i + 0.5, name, ha='left', va='center', fontsize=8)

plt.show()

# Save each color as a square PNG image
for i, color in enumerate(sorted_rgb_colors):
    # Create a 100x100 pixel image with the current color
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:] = (color * 255).astype(np.uint8)
    # Create a valid filename from the color name
    filename = f"{custom_color_names[i].replace(' ', '_')}.png"
    plt.imsave(filename, img)