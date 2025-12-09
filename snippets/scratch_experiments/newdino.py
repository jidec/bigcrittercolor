import cv2
import numpy as np
from PIL import Image
import torch
from typing import Union, Tuple, List, Optional
from huggingface_hub import hf_hub_download

from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import annotate, load_image, predict


def detect_insects_on_sheet(
        image_path: Union[str, np.ndarray],
        text_prompt: str = "insect",
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        use_gpu: bool = True,
        model_repo: str = "ShilongLiu/GroundingDINO",
        model_file: str = "groundingdino_swinb_cogcoor.pth",
        config_file: str = "GroundingDINO_SwinB.cfg.py",
        visualize: bool = True,
        save_visualization: Optional[str] = None,
        return_image: bool = False,
        verbose: bool = True
) -> Union[Tuple[torch.Tensor, torch.Tensor, List[str]],
Tuple[torch.Tensor, torch.Tensor, List[str], np.ndarray]]:
    """
    Detect insects on a moth lighttrapping sheet using GroundingDINO.

    Args:
        image_path: Path to image file or cv2 image array (BGR format)
        text_prompt: Text prompt for detection (default: "insect")
        box_threshold: Confidence threshold for keeping bounding boxes (0-1)
                      Higher = fewer, more confident detections
        text_threshold: Confidence threshold that boxes match the prompt (0-1)
        use_gpu: Whether to use GPU acceleration if available
        model_repo: HuggingFace repository ID for GroundingDINO
        model_file: Model checkpoint filename
        config_file: Model configuration filename
        visualize: Whether to display the annotated image
        save_visualization: Path to save visualization (if provided)
        return_image: Whether to return the annotated image array
        verbose: Whether to print progress messages

    Returns:
        boxes: Tensor of bounding boxes in cxcywh format (normalized 0-1)
        logits: Tensor of confidence scores for each box
        phrases: List of detected class labels for each box
        annotated_image: (optional) Annotated image array if return_image=True

    Example:
        >>> boxes, scores, labels = detect_insects_on_sheet(
        ...     "moth_sheet.jpg",
        ...     box_threshold=0.35,
        ...     visualize=True
        ... )
        >>> print(f"Detected {len(boxes)} insects")
        >>> print(f"Confidence scores: {scores}")
    """

    # Load image
    if verbose:
        print("Loading image...")

    if isinstance(image_path, str):
        image_source, image = load_image(image_path)
        if image_source is None:
            raise ValueError(f"Failed to load image from {image_path}")
    elif isinstance(image_path, np.ndarray):
        # Save temporarily and load (to match expected format)
        temp_path = "/tmp/temp_insect_detection.jpg"
        cv2.imwrite(temp_path, image_path)
        image_source, image = load_image(temp_path)
    else:
        raise TypeError("image_path must be a file path string or numpy array")

    # Load GroundingDINO model
    if verbose:
        print("Loading GroundingDINO model...")

    model = _load_grounding_dino(
        repo_id=model_repo,
        filename=model_file,
        config_filename=config_file,
        use_gpu=use_gpu
    )

    # Detect insects
    if verbose:
        print(f"Detecting '{text_prompt}' with box_threshold={box_threshold}, "
              f"text_threshold={text_threshold}...")

    device = 'cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu'

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device
    )

    num_detections = boxes.shape[0] if boxes.nelement() > 0 else 0
    if verbose:
        print(f"Detected {num_detections} insects")

    # Visualize if requested
    annotated_image = None
    if visualize or save_visualization or return_image:
        if num_detections > 0:
            annotated_frame = annotate(
                image_source=image_source,
                boxes=boxes,
                logits=logits,
                phrases=phrases
            )
            # Convert BGR to RGB for display
            annotated_frame = annotated_frame[..., ::-1]
            annotated_image = annotated_frame

            if visualize:
                img_pil = Image.fromarray(annotated_frame)
                img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

                # Display with window that can be resized
                window_name = f"Insect Detection - {num_detections} insects found"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow(window_name, img_cv)
                print("Press any key to close visualization window...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if save_visualization:
                img_pil = Image.fromarray(annotated_frame)
                img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_visualization, img_cv)
                if verbose:
                    print(f"Saved visualization to {save_visualization}")
        else:
            if verbose:
                print("No insects detected - skipping visualization")

    # Return results
    if return_image:
        return boxes, logits, phrases, annotated_image
    else:
        return boxes, logits, phrases


def _load_grounding_dino(repo_id: str, filename: str, config_filename: str,
                         use_gpu: bool = True):
    """Load GroundingDINO model from HuggingFace Hub."""

    device = 'cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu'

    # Download config and weights
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=config_filename)
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)

    # Build model
    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device

    # Load weights
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print(f"Model loaded from {cache_file}")

    model.eval()
    if use_gpu and torch.cuda.is_available():
        model = model.to(device)

    return model


def get_bbox_xyxy(boxes: torch.Tensor, image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert normalized cxcywh boxes to pixel xyxy format.

    Args:
        boxes: Tensor of boxes in cxcywh format (normalized 0-1)
        image_shape: (height, width) of the image

    Returns:
        Array of boxes in xyxy pixel format: [[x1, y1, x2, y2], ...]
    """
    H, W = image_shape[:2]
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    return boxes_xyxy.cpu().numpy()

if __name__ == "__main__":
    # Basic usage
    boxes, scores, labels = detect_insects_on_sheet(
        "D:/moth_test_imgs/ama_2025-08-11_22_30_02.jpg",
        box_threshold=0.30,
        text_threshold=0.25,
        visualize=True
    )

    print(f"\nDetection Summary:")
    print(f"Total insects detected: {len(boxes)}")
    print(f"Confidence scores: {scores.cpu().numpy()}")
    print(f"Labels: {labels}")

    # Save visualization and get bounding box coordinates
    boxes, scores, labels, annotated_img = detect_insects_on_sheet(
        "moth_lightsheet.jpg",
        save_visualization="output_annotated.jpg",
        return_image=True,
        verbose=True
    )

    # Get boxes in pixel coordinates (xyxy format)
    if len(boxes) > 0:
        H, W = annotated_img.shape[:2] if annotated_img is not None else (0, 0)
        if H > 0:
            boxes_xyxy = get_bbox_xyxy(boxes, (H, W))
            print(f"\nBounding boxes (x1, y1, x2, y2):")
            for i, box in enumerate(boxes_xyxy):
                print(f"  Insect {i + 1}: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}] "
                      f"- confidence: {scores[i]:.3f}")