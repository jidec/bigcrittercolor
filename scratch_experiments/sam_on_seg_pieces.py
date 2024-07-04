import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

from bigcrittercolor.helpers import _readBCCImgs
import random

imgs = _readBCCImgs(type="raw_segment", data_folder="D:/dfly_appr_expr/appr1")
imgs = random.sample(imgs,100)

import sys

sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "D:/dfly_appr_expr/appr1/other/ml_checkpoints/sam.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

all_segs = []
for image in imgs:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print('gen image')
    masks = mask_generator.generate(image)

    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.show()

    img = np.zeros_like(image)
    segs_px = [mask['segmentation'] for mask in masks]
    #segs = []
    for seg_px in segs_px:
        img = np.zeros_like(image)
        img[seg_px] = (255,255,255)
        #cv2.imshow("0",img)
        #cv2.waitKey(0)
        all_segs.append(img)

print("done getting all segs")

from bigcrittercolor.helpers.imgtransforms import _verticalizeImg
from bigcrittercolor.helpers import _clusterByImgFeatures

all_segs = [_verticalizeImg(seg) for seg in all_segs]

print("done vert")

_clusterByImgFeatures(all_segs, feature_extractor="resnet18")

# vert them
# cluster them
# should take an hour or so for like 100
