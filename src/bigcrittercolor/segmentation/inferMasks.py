import os, sys
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
from PIL import Image, ImageDraw, ImageFont
# Grounding DINO
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate, load_image, predict
# segment anything
from segment_anything import build_sam, SamPredictor
import cv2
import numpy as np
# diffusers
import torch
from huggingface_hub import hf_hub_download
from bigcrittercolor.helpers import _bprint, _getIDsInFolder, _showImages
import random

def inferMasks(img_ids, skip_existing=True,
               strategy="prompt1", text_prompt="subject", box_threshold=0.25, text_threshold=0.25,
               erode_kernel_size=0, remove_islands = True,
               show=False, show_indv=False, print_steps=True, print_details=False, data_folder=""):
    """ Infer masks for the specified image ids using SegmentAnything prompted by the groundingDINO zero-shot object detector

        Args:
            img_ids (list): the imageIDs (image names) to infer from
            skip_existing (bool): if we skip images that have already been processed
            strategy (str): prepackaged strategies for creating masks from iNat images
                can be:
                1. prompt1 - simply save the first mask of prompt1 - a reliable way to get the image subject
                2. remove_prompt2_from1 - save prompt1 with all instances of prompt2 (+3,4,etc. currently) subtracted

                to add later:
                3. prompt1_merged_and_prompt2_merged - save all instances of prompt1 merged together, AND all instances of prompt2 merged
                4. prompt1_instances - save all instances of prompt1 separately
                5. prompt1_instances_and_prompt2_instances - save all instances of prompt1 and
            text_prompt (str): the text prompt for groundingDINO to use in selecting bounding boxes
            box_threshold (float): confidence threshold to keep object bounding boxes - higher values means less boxes are kept
            text_threshold (float): confidence threshold that kept object bounding boxes are the object we prompted for
            erode_kernel_size (int): size of the postprocessing erosion kernel
            remove_islands (bool): whether to remove islands from the mask during postprocessing
    """

    # download grounding dino if you need it, then load it
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

    _bprint(print_steps,"Loading groundingDINO, downloading weights if necessary...")
    groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)

    # sam currently must be downloaded from https://huggingface.co/ybelkada/segment-anything/tree/main/checkpoints and placed in
    #   the ml_checkpoints folder as sam.pth
    _bprint(print_steps, "Loading SegmentAnything from data_folder/other/ml_checkpoints/sam.pth...")
    sam_checkpoint = data_folder + '/other/ml_checkpoints/sam.pth'
    sam = build_sam(checkpoint=sam_checkpoint)
    sam.to(device='cuda')
    sam_predictor = SamPredictor(sam)

    # if no ids get all existing
    if img_ids is None:
        _bprint(print_steps, "No IDs specified, getting all existing image IDs from all_images")
        img_ids = _getIDsInFolder(data_folder + "/all_images")
        random.shuffle(img_ids)


    if skip_existing:
        # get existing mask ids
        existing_mask_ids = _getIDsInFolder(data_folder + "/masks")
        # create new image ids removing existing
        img_ids = [item for item in img_ids if item not in existing_mask_ids]

    # create img_locs which starts as a copy of img_ids
    img_locs = img_ids.copy()
    # turn list of ids into list of file locations
    for i in range(0,len(img_locs)):
        img_locs[i] = data_folder + "/all_images/" + img_locs[i] + ".jpg"

    _bprint(print_steps, "Starting inferring masks for " + str(len(img_locs)) + " IDs...")
    # for each image location
    for index, img_loc in enumerate(img_locs):

        _bprint(print_details, "Reading in image from " + img_loc + "...")
        image_source, image = load_image(img_loc)

        if (image_source is None):
            _bprint(print_details, "Image load failed, skipping...")
            continue

        if show_indv:
            img_raw = cv2.imread(img_loc)
            #_showImages(show=True, images=[cv2.imread(img_loc)], titles=["Raw Image"])

        _bprint(print_details, "Getting bounding boxes using groundingDINO...")
        boxes, logits, phrases = predict(
            model=groundingdino_model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )

        if boxes.nelement() == 0:
            _bprint(print_details, "No bounding boxes obtained, skipping...")
            continue

        if show_indv:
            annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
            annotated_frame = annotated_frame[..., ::-1]  # BGR to RGB
            img = Image.fromarray(annotated_frame)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            #_showImages(show=True, images=[img], titles=["Annotated Frames"])
            img_frame = np.copy(img)

        _bprint(print_details, "Predicting mask using SegmentAnything...")

        # set image
        sam_predictor.set_image(image_source)

        # box: normalized box xywh -> unnormalized xyxy
        H, W, _ = image_source.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to('cuda:0')
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        if show_indv:
            annotated_frame_with_mask = show_mask(masks[0][0], annotated_frame)
            img = Image.fromarray(annotated_frame_with_mask)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img_frame_mask = np.copy(img)
            #_showImages(show=True, images=[img],titles=["Annotated Frames with Mask"])

        # if we didn't get any segments, skip
        if masks[0][0].cpu().nelement() == 0:
            _bprint(print_details, "No masks predicted, skipping...")
            continue

        _bprint(print_details, "Applying strategy...")
        # this strategy simply saves the first mask
        if strategy == "prompt1":
            mask = tensor_to_img(masks[0][0])

        # this strategy saves the first mask with ALL subsequent masks removed from it
        if strategy == "remove_prompt2_from1":

            # get the first (prompt1) mask
            mask = tensor_to_img(masks[0][0])
            #cv2.imshow("prompt1_mask", mask)
            #cv2.waitKey(0)

            # if more than 1 mask, collate subsequent masks
            if len(masks) > 1:
                prompt2_mask = tensor_to_img(masks[1])

                #cv2.imshow("class2_init", class2_mask)
                #cv2.waitKey(0)

                # for each mask of class 2
                for i in range(2,len(masks)):
                    mask_to_add = tensor_to_img(masks[i])
                    #cv2.imshow("class2_to_add", mask_to_add)
                    #cv2.waitKey(0)
                    prompt2_mask = np.logical_or(prompt2_mask, mask_to_add).astype(np.uint8) * 255
                    #cv2.imshow("class2_with_add", prompt2_mask)
                    #cv2.waitKey(0)

                #cv2.imshow("1", prompt2_mask)
                #cv2.waitKey(0)

                # remove the prompt2_mask from the original prompt 1 mask
                # taking the AND of the prompt 1 mask and the NOT prompt2_mask accomplishes this
                mask = cv2.bitwise_and(mask, cv2.bitwise_not(prompt2_mask))

            #cv2.imshow("final", mask)
            #cv2.waitKey(0)

        # if mask is all black (empty) skip
        if cv2.countNonZero(mask) == 0:
            _bprint(print_details, "Empty mask, skipping...")
            continue

        if show_indv:
            mask_raw = np.copy(mask)
            #_showImages(show=True, images=[mask], titles=["Mask"])

        # erode mask
        if erode_kernel_size > 0:
            _bprint(print_details, "Eroding mask...")
            kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
            mask = cv2.erode(mask, kernel)
            if show_indv:
                mask_eroded = np.copy(mask)
                #_showImages(show=True, images=[mask], titles=["Eroded Mask"])

        # keep only the largest component in the image, removing tiny blips
        if remove_islands:
            # keep only biggest contour (i.e. remove islands from mask)
            _bprint(print_details, "Removing islands from mask...")
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # if no contour found, skip
            if not contours:
                _bprint(print_details, "No contour found when removing islands, skipping...")
                continue

            # empty image and fill with big contour
            mask = np.zeros_like(mask)
            cv2.drawContours(mask, [max(contours, key=len)], -1, 255, thickness=-1)

            if show_indv:
                mask_islands = np.copy(mask)
                raw_segment = cv2.bitwise_and(img_raw, img_raw, mask=mask.astype(np.uint8))
                _showImages(show=True, images=[img_raw,img_frame,img_frame_mask, mask_eroded, mask_islands, raw_segment],
                            titles=["Start Image", "Image w/ Frame", "Image with Mask", "Eroded Mask", "Islands-Removed Mask", "Raw Segment"],
                            title_fontsize=13)

        _bprint(print_details, "Writing final mask...")
        dest = data_folder + "/masks/" + img_ids[index] + "_mask.png"
        cv2.imwrite(dest, mask)

    # if show, show a sample of the masks we got
    if show:
        masks = [cv2.imread(data_folder + "/masks/" + id + "_mask.png") for id in img_ids]
        masks = random.sample(masks,12)
        _showImages(show,masks,maintitle="Sample Masks")


# download groundingDINO if you need it, the  load it
def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model

# parse masks from tensors of bools to cv2 images
def tensor_to_img(t):
    t = t.cpu()
    h, w = t.shape[-2:]
    mask = (t.reshape(h, w, 1).numpy() * 255).astype(np.uint8)
    return (mask)

# show the SAM mask
def show_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    mask = mask.cpu()
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))