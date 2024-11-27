import cv2
import numpy as np
from PIL import Image
import random

# Grounding DINO
#sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import annotate, load_image, predict

# SAM
from segment_anything import build_sam, SamPredictor

import torch
from huggingface_hub import hf_hub_download

from bigcrittercolor.helpers import _bprint, _showImages, _getBCCIDs,_writeBCCImgs,_readBCCImgs
from bigcrittercolor.helpers.verticalize import _verticalizeImg
from bigcrittercolor.helpers.image import _removeIslands, _imgAndMaskAreValid

def inferMasks2(img_ids=None, skip_existing=True, gd_gpu=True, sam_gpu=True,
               segmentation_actions=None,
               strategy="prompt1", text_prompt="subject", box_threshold=0.25, text_threshold=0.25,
               aux_segmodel_location=None,
               auxseg_normalize_params_dict={'lines_strategy':"skeleton_hough", 'best_line_metric':"overlap_sym"},
               sam_location=None,
               erode_kernel_size=0, remove_islands=True,
               show=False, show_indv=False, print_steps=True, print_details=False, data_folder=""):

    # Load groundingDINO model
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filename = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

    _bprint(print_steps, "Loading groundingDINO, downloading weights if necessary...")
    groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename)

    # Load SAM model
    _bprint(print_steps, "Loading SegmentAnything...")
    if sam_location is None:
        sam_checkpoint = data_folder + '/other/ml_checkpoints/sam.pth'
    else:
        sam_checkpoint = sam_location
    sam = build_sam(checkpoint=sam_checkpoint)
    if sam_gpu:
        sam.to(device='cuda')
    sam_predictor = SamPredictor(sam)

    # Process image IDs
    if img_ids is None:
        _bprint(print_steps, "No IDs specified, getting all existing image IDs from all_images")
        img_ids = _getBCCIDs(type="image", data_folder=data_folder)

    if skip_existing:
        existing_mask_ids = _getBCCIDs(type="mask", data_folder=data_folder)
        with open(data_folder + "/other/processing_info/failed_mask_infers.txt") as file:
            failed_mask_ids = [line.strip() for line in file]
        existing_mask_ids += failed_mask_ids
        img_ids = list(set(img_ids) - set(existing_mask_ids))

    if aux_segmodel_location is not None:
        aux_model = torch.load(aux_segmodel_location)
        aux_model.eval()
        # pick cuda device
        aux_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def write_failed_id(id):
        with open(data_folder + "/other/processing_info/failed_mask_infers.txt") as file:
            ids = [line.strip() for line in file]
        ids.append(id)
        ids = list(set(ids))
        with open(data_folder + "/other/processing_info/failed_mask_infers.txt", 'w') as file: file.write('\n'.join(ids))

    def aux_model(image_source, text_prompt, box_threshold=0.25, text_threshold=0.25, remove_islands=True, show_indv=False):
        _bprint(print_details, "Applying auxiliary model to SAM segment...")

        # save dims of vert'ed SAM segment
        image_source_dims = (image_source.shape[1], image_source.shape[0])

        # make tensor to feed into NN
        image_source_tensor = np.copy(image_source)
        image_source_tensor = cv2.resize(image_source_tensor, (344, 344))  # 344 is the size the NN takes

        # transpose to correct shape for model
        image_source_tensor = image_source_tensor.transpose(2, 0, 1).reshape(1, 3, 344, 344)

        # model to device
        aux_model.to(aux_device)

        # get output from model
        with torch.no_grad():
            a = aux_model((torch.from_numpy(image_source_tensor).type(torch.cuda.FloatTensor) / 255).to(aux_device))

        activation_threshold = 0.4
        # make mask using activation threshold
        mask = a['out'].cpu().detach().numpy()[0][0] > activation_threshold
        mask = mask.astype(np.uint8)  # convert to an unsigned byte
        mask *= 255

        # resize to the original input image size
        mask = cv2.resize(mask, image_source_tensor_dims)

        # remove islands
        mask = mask.astype(np.uint8)
        mask = _removeIslands(mask)

        # apply the new mask to get the seg
        seg_after_aux = cv2.bitwise_and(image_source, image_source, mask=mask.astype(np.uint8))

        # reverticalize the new seg
        seg_after_aux = _verticalizeImg(seg_after_aux, lines_strategy="ellipse")
        show_indv_data.append((seg_after_aux, "Seg after Aux"))

        # even though it's not a mask, let's call it one so that it gets written the same way as the normal approach
        mask = seg_after_aux

        # show the individual process
        _showImages(show=show_indv, images=[t[0] for t in show_indv_data],
                    titles=[t[1] for t in show_indv_data],
                    title_fontsize=13)
    # Function for groundedSAM processing
    def grounded_sam(image_source, text_prompt, box_threshold=0.25, text_threshold=0.25, remove_islands=True, show_indv=False):
        original_img = image_source
        _bprint(print_details, "Getting bounding boxes using groundingDINO...")
        # NOTE that groundingDINO predict here is throwing a cpu error - cuda is not installed
        if gd_gpu:
            boxes, logits, phrases = predict(
                model=groundingdino_model,
                image=image,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )
        else:
            boxes, logits, phrases = predict(
                model=groundingdino_model,
                image=image,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device='cpu'
            )
        if boxes.nelement() == 0:
            _bprint(print_details, "No bounding boxes obtained, skipping...")
            write_failed_id(img_ids[index])
            return None

        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        annotated_frame = annotated_frame[..., ::-1]  # BGR to RGB
        img = Image.fromarray(annotated_frame)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_frame = np.copy(img)

        _bprint(print_details, "Predicting mask using SegmentAnything...")

        # set image
        sam_predictor.set_image(image_source)

        # box: normalized box xywh -> unnormalized xyxy
        H, W, _ = image_source.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2])
        if sam_gpu:
            transformed_boxes = transformed_boxes.to('cuda:0')
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        annotated_frame_with_mask = show_mask(masks[0][0], annotated_frame)
        img = Image.fromarray(annotated_frame_with_mask)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_frame_mask = np.copy(img)

        # if we didn't get any segments, skip
        if masks[0][0].cpu().nelement() == 0:
            _bprint(print_details, "No masks predicted, skipping...")
            write_failed_id(img_ids[index])
            return(None)

        mask = tensor_to_img(masks[0][0])

        # if mask is all black (empty) skip
        if cv2.countNonZero(mask) == 0:
            _bprint(print_details, "Empty mask, skipping...")
            write_failed_id(img_ids[index])
            return(None)

        _showImages(True,[original_img,img_frame,img_frame_mask])

        img = cv2.bitwise_and(original_img, original_img, mask=mask)
        #show_indv_data = [(img_raw,"Start Image"),(img_frame,"Image w/ Frame"),(img_frame_mask,"Image with Mask")]

        return img

    # Default segmentation actions
    if segmentation_actions is None:
        segmentation_actions = {
            "groundedSAM_subject": {
                "model": "groundingDINO",
                "action": "get",
                "text_prompt": "insect"
            },
            "normalize1": {
                "model": "verticalize"
            },
            "groundedSAM_tail": {
                "model": "groundingDINO",
                "action": "get",
                "text_prompt": "tail . insect"
            }
        }

    _bprint(print_steps, "Starting mask inference for {} IDs...".format(len(img_ids)))

    for index, id in enumerate(img_ids):
        img = _readBCCImgs(type="image", img_ids=id, data_folder=data_folder)
        img_loc = data_folder + "/other/temp.jpg"
        cv2.imwrite(filename=img_loc, img=img)
        _bprint(print_details, "Reading image {}...".format(id))

        image_source, image = load_image(img_loc)

        if image_source is None:
            _bprint(print_details, "Image load failed, skipping...")
            write_failed_id(id)
            continue

        img_raw = cv2.imread(img_loc)

        iterated_img = img_raw
        # Loop over segmentation actions
        for action_name, action_params in segmentation_actions.items():
            _bprint(print_details, f"Performing action: {action_name}")

            if action_params["model"] == "groundingDINO":
                text_prompt = action_params["text_prompt"]
                iterated_img = grounded_sam(iterated_img, text_prompt, box_threshold, text_threshold, remove_islands, show_indv)
                cv2.imshow("0",iterated_img)
                cv2.waitKey(0)
                if iterated_img is None:
                    write_failed_id(id)
                    continue
            if action_params["model"] == "verticalize":
                iterated_img = _verticalizeImg(iterated_img)

    if show:
        masks = _readBCCImgs(type="mask", sample_n=12, data_folder=data_folder)
        _showImages(show, masks, maintitle="Sample Masks")

# Helper function for loading models
def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    model.eval()
    return model

# Helper function for converting tensors to images
def tensor_to_img(t):
    t = t.cpu()
    h, w = t.shape[-2:]
    mask = (t.reshape(h, w, 1).numpy() * 255).astype(np.uint8)
    return mask

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

inferMasks2(skip_existing=False,data_folder="D:/bcc/ringtails_copy",show_indv=True,show=True,sam_location="D:/bcc/sam.pth")