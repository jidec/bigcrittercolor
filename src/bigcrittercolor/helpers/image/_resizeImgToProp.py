import cv2

def _resizeImgToProp(img, prop):
    # Calculate the new dimensions based on the proportion
    new_width = int(img.shape[1] * prop)
    new_height = int(img.shape[0] * prop)

    # Resize the image
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return resized_img
    