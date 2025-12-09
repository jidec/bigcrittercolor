import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, deltaE_cie76
from bigcrittercolor.helpers.image import _blur

def region_growing(image, seeds, initial_threshold=10):
    height, width, channels = image.shape
    processed = np.zeros((height, width), dtype=bool)
    regions = []

    def get_neighbors(x, y):
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                neighbors.append((nx, ny))
        return neighbors

    def color_distance(c1, c2):
        # Use deltaE_cie76 for better color difference measurement
        lab1 = rgb2lab(np.uint8([[c1]]))
        lab2 = rgb2lab(np.uint8([[c2]]))
        return deltaE_cie76(lab1, lab2)[0][0]

    for seed in seeds:
        x, y = seed
        if processed[y, x]:
            continue
        region = []
        stack = [(x, y)]
        seed_color = image[y, x]
        while stack:
            px, py = stack.pop()
            if processed[py, px]:
                continue
            processed[py, px] = True
            region.append((px, py))
            current_threshold = initial_threshold
            for nx, ny in get_neighbors(px, py):
                if not processed[ny, nx]:
                    neighbor_color = image[ny, nx]
                    if color_distance(neighbor_color, seed_color) < current_threshold:
                        stack.append((nx, ny))
                    else:
                        # If not similar enough, reduce the threshold to capture fine details
                        current_threshold = initial_threshold / 2
                        if color_distance(neighbor_color, seed_color) < current_threshold:
                            stack.append((nx, ny))
        if region:
            regions.append(region)

    return regions


# Load the image
image_path = 'D:/bcc/ringtails/segments/INAT-147315-1_segment.png'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = _blur(image,type="bilateral")

# Define seeds (using a denser grid-based approach)
height, width, _ = image.shape
num_seeds = 40  # Increase the number of seeds for finer details
grid_size = max(height, width) // num_seeds
seeds = [(x, y) for x in range(0, width, grid_size) for y in range(0, height, grid_size)]

# Perform region growing
regions = region_growing(image, seeds, initial_threshold=20)

# Visualize the regions
output_image = np.zeros_like(image)
for region in regions:
    color = np.random.randint(0, 255, size=3)
    for (x, y) in region:
        output_image[y, x] = color

# Display the original and segmented images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Region Growing Result')
plt.imshow(output_image)
plt.axis('off')

plt.show()