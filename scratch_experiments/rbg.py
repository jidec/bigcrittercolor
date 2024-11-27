import cv2

# Load the image
image = cv2.imread("C:/Users/hiest/Desktop/Ruth_Bader_Ginsburg_2016_portrait.jpg")

# Resize the image to 0.25x its original size
height, width = image.shape[:2]
resized_image = cv2.resize(image, (int(width * 0.25), int(height * 0.25)))

# Swap green and blue channels
swapped_image = resized_image.copy()
swapped_image[:, :, 0], swapped_image[:, :, 1] = resized_image[:, :, 1], resized_image[:, :, 0]

# Display the original and modified images
cv2.imshow('Original Image', image)
cv2.imshow('Resized and Swapped Image', swapped_image)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()