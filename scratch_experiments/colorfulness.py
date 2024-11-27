import cv2
import numpy as np


def colorfulness_index(image):
    # Split the image into its RGB components
    (B, G, R) = cv2.split(image.astype("float"))

    # Create a mask to select only non-black pixels (where all channels are not zero)
    mask = (R != 0) | (G != 0) | (B != 0)

    # Apply the mask to the R, G, and B channels
    R_non_black = R[mask]
    G_non_black = G[mask]
    B_non_black = B[mask]

    # Compute the rg and yb components for non-black pixels
    rg = np.abs(R_non_black - G_non_black)
    yb = np.abs(0.5 * (R_non_black + G_non_black) - B_non_black)

    # Compute the mean and standard deviation of both rg and yb
    rg_mean, rg_std = np.mean(rg), np.std(rg)
    yb_mean, yb_std = np.mean(yb), np.std(yb)

    # Combine the mean and standard deviation
    std_root = np.sqrt((rg_std ** 2) + (yb_std ** 2))
    mean_root = np.sqrt((rg_mean ** 2) + (yb_mean ** 2))

    # The colorfulness index
    colorfulness = std_root + (0.3 * mean_root)

    return colorfulness


from bigcrittercolor.helpers import _readBCCImgs

# Example usage
#image = cv2.imread("D:/bcc/new_random_dragonflies/segments/INATRANDOM-29363191_segment.png")  # Replace with the actual image path
imgs = _readBCCImgs(type="segment",data_folder="D:/bcc/ringtails")

for img in imgs:
    cv2.imshow("0",img)
    cv2.waitKey(0)
    colorfulness = colorfulness_index(img)
    print("Colorfulness Index:", colorfulness)
