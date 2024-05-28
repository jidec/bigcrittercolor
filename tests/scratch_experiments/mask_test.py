import numpy as np
import matplotlib.pyplot as plt

# Creating a synthetic image (100x100 pixels) with a mix of pure black and other colors
# Initialize the image with random colors
image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

# Add a pure black square in the center
image[40:60, 40:60, :] = 0

# Generate a mask for pure black pixels
# This creates a boolean mask where True represents a pure black pixel
mask = np.all(image == [0, 0, 0], axis=-1)

# Visualize the original image and the mask
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Show the original image
ax[0].imshow(image)
ax[0].set_title('Original Image')
ax[0].axis('off')

# Show the mask (convert boolean mask to an image for visualization)
ax[1].imshow(mask, cmap='gray')
ax[1].set_title('Mask for Pure Black Areas')
ax[1].axis('off')

plt.show()