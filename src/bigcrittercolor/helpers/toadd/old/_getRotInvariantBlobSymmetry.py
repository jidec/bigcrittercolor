import cv2
import numpy as np
from bigcrittercolor.helpers import _showImages

# this function returns the symmetry score by default BUT can also return the rotated image for
# use in verticalizeImg
def _getRotInvariantBlobSymmetry(binary_img, steps=30, return_sym_img=False, show=False):
    # Compute the centroid of the blob
    M = cv2.moments(binary_img)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    if show:
        centroid_img = np.copy(binary_img)
        centroid_img = cv2.circle(centroid_img, (cX,cY), 30, (0), -1)
        _showImages(show,centroid_img,maintitle="Image with Centroid")

    # Use a larger canvas to ensure the blob remains within boundaries during rotations
    max_dim = max(binary_img.shape) * 2
    canvas = np.zeros((max_dim, max_dim), dtype=np.uint8)
    offsetX, offsetY = (max_dim - binary_img.shape[1]) // 2, (max_dim - binary_img.shape[0]) // 2
    canvas[offsetY:offsetY + binary_img.shape[0], offsetX:offsetX + binary_img.shape[1]] = binary_img

    # Update centroid for the larger canvas
    cX += offsetX
    cY += offsetY

    max_symmetry = 0
    best_angle = 0
    #steps = 60  # Since we're measuring symmetry, we only need to check up to 180 degrees
    steps_since_show = 5
    for angle in np.linspace(0, 180, steps):
        # Rotate the blob around the centroid
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1)
        rotated = cv2.warpAffine(canvas, M, (max_dim, max_dim), flags=cv2.INTER_NEAREST)

        left_side = rotated[:, :cX]
        right_side = np.fliplr(rotated[:, cX:])

        # Measure overlap in the "folded" parts
        min_width = min(left_side.shape[1], right_side.shape[1])
        overlap = np.sum(np.bitwise_and(left_side[:, -min_width:], right_side[:, :min_width]))

        symmetry_score = overlap / (np.sum(left_side) + np.sum(right_side))

        if symmetry_score > max_symmetry:
            max_symmetry = symmetry_score
            best_angle = angle
            if steps_since_show >= 5:
                steps_since_show = 0
                #_showImages(show, [left_side, right_side], maintitle="Sym Sides")
        else:
            steps_since_show = steps_since_show + 1

    # Rotate the blob around the centroid
    M = cv2.getRotationMatrix2D((cX, cY), best_angle, 1)
    best_sym_img = cv2.warpAffine(canvas, M, (max_dim, max_dim), flags=cv2.INTER_NEAREST)
    _showImages(show, best_sym_img, maintitle="Best Sym Angle")

    if return_sym_img:
        return best_sym_img

    return max_symmetry