import cv2

# convert a seg to a mask by setting nonblack pixels to white
def _segToMask(rgb_img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image to create a binary mask.
    # Pixels with value > 0 will be set to 255 (white) and pixels with value 0 remain black.
    _, binary_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    return binary_mask