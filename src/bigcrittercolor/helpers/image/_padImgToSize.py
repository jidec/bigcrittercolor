import cv2

def _padImgToSize(img, target_wh):
    target_width = target_wh[0]
    target_height = target_wh[1]

    if img.shape[0] > target_height or img.shape[1] > target_width:
        print("Target dimensions are smaller than the original image. Cannot pad.")
        return

    # Determine the amount of padding needed for height and width
    top_padding = (target_height - img.shape[0]) // 2
    bottom_padding = target_height - img.shape[0] - top_padding

    left_padding = (target_width - img.shape[1]) // 2
    right_padding = target_width - img.shape[1] - left_padding

    # Apply padding
    padded_img = cv2.copyMakeBorder(img, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return padded_img