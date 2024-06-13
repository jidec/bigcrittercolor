from bigcrittercolor.helpers.image import _imgToColorPatches
import cv2


img = cv2.imread('E:/aeshna_data/segments/INAT-7656493-4_segment.png')
img = cv2.imread('E:/aeshna_data/segments/INAT-7444110-10_segment.png')
img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

img2 = _imgToColorPatches(img)
img = _imgToColorPatches(img,return_patch_pixels_and_color=True)
cv2.imshow('0',img2)
cv2.waitKey(0)
print(img)
print(len(img[0]))
print(len(img[1]))

# Assuming 'masks' is a list of boolean masks and 'colors' is a list of colors
masks = img[0] # Replace with your list of masks
colors = img[1]  # Replace with your list of colors (e.g., [(255, 0, 0), (0, 255, 0), ...])

# Create an empty image, assuming all masks are of the same size
height, width = masks[0].shape
reconstructed_image = np.zeros((height, width, 3), dtype=np.uint8)

# Iterate over each mask and color, and apply the color to the masked region
for mask, color in zip(masks, colors):
    # The mask is used to find the coordinates where the color should be applied
    reconstructed_image[mask] = color

# Display the image or save it
cv2.imshow('Reconstructed Image', reconstructed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

_showImages._showImages(True,[img2,reconstructed_image])