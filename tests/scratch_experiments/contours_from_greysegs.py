import cv2
import numpy as np

from bigcrittercolor.helpers import _getIDsInFolder

ids = _getIDsInFolder("D:/bcc/dfly_appr_expr/appr1/masks")
ids = ids[23:]
for id in ids:
    print(id)
    loc = "D:/bcc/dfly_appr_expr/appr1/masks/" + id + "_mask.png"
    img = cv2.imread(loc, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    src_img = cv2.imread("D:/bcc/dfly_appr_expr/appr1/all_images/" + id + ".jpg")
    src_img = cv2.resize(src_img, (src_img.shape[1] // 2, src_img.shape[0] // 2))

    masked = cv2.bitwise_and(src_img, src_img, mask=img.astype(np.uint8)) #mask=cv2.cvtColor(line_img,cv2.COLOR_RGB2GRAY).astype(np.uint8))

    cv2.imshow("0",masked)
    cv2.waitKey(0)

    # Convert the image to grayscale
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

    cv2.imshow("0", gray)
    cv2.waitKey(0)

    #gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, threshold1=25, threshold2=100)
    #sobelxy = cv2.Sobel(src=gray, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection

    cv2.imshow("0", edges)
    cv2.waitKey(0)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the original image
    # -1 means drawing all contours; (0, 255, 0) specifies the color (green in this case); 2 is the line thickness
    cont_img = np.zeros_like(img)
    cv2.drawContours(cont_img, contours, -1, (0, 255, 0), 2)

    # Show the image with contours
    cv2.imshow('Contours', cont_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()