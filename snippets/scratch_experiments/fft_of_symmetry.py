import cv2
import numpy as np
import matplotlib.pyplot as plt

from bigcrittercolor.helpers import _getIDsInFolder

ids = _getIDsInFolder("D:/bcc/dfly_appr_expr/appr1/masks")
ids = ids[27:]
for id in ids:
    print(id)
    loc = "D:/bcc/dfly_appr_expr/appr1/masks/" + id + "_mask.png"

    # Load the image and convert to grayscale
    image = cv2.imread(loc, 0)

    # Compute the 2D Fourier Transform of the image
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    # Compute magnitude spectrum
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    # Visualize magnitude spectrum
    plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()