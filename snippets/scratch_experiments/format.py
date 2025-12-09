import cv2
import numpy as np

seg = cv2.imread("D:/bcc/my_taxon/segments/INAT-69225192-3_segment.png")
mask = cv2.imread("D:/bcc/my_taxon/masks/INAT-69225192-3_mask.png")
img = cv2.imread("D:/bcc/my_taxon/all_images/INAT-69225192-3.jpg")

seg[np.where((seg == [0, 0, 0]).all(axis=2))] = [0, 0, 255]

nat_seg = cv2.bitwise_and(img, img, mask=cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY).astype(np.uint8))
nat_seg[np.where((nat_seg == [0, 0, 0]).all(axis=2))] = [0, 0, 255]

# you can see that in the seg edges are getting fuzzy (light black) whereas in the nat seg their are clean
cv2.imshow('0',seg)
cv2.waitKey(0)
cv2.imshow('0',nat_seg)
cv2.waitKey(0)
# what transforms have ahppened to cause this? in Clusterextract segs

# lets double check that verticalize is causing the problem
# YUP IT IS, with default params but ellipse lines

# lets vert the nat seg
from bigcrittercolor.helpers.verticalize import _verticalizeImg
nat_seg = cv2.bitwise_and(img, img, mask=cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY).astype(np.uint8))


#nat_seg = cv2.resize(nat_seg , None, fx=3.0, fy=3.0, interpolation=cv2.INTER_NEAREST)

vert = _verticalizeImg(nat_seg)
vert[np.where((vert == [0, 0, 0]).all(axis=2))] = [0, 0, 255]

cv2.imshow("5",vert)
cv2.waitKey(0)
# its in vertUsingLine

# edges are fixed, now the problem is in clusterColorsToPatterns, I think in imgsToPatches
