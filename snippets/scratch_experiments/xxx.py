import cv2
import numpy as np
from PIL import Image
import torch
import torchvision
from typing import Union, Tuple, List, Optional
from huggingface_hub import hf_hub_download

from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import annotate, predict
import groundingdino.datasets.transforms as T


def calculate_sliding_windows(
        image_shape: Tuple[int, int],
        num_windows: int = 3,
        target_short_side: int = 800,
        target_long_side: int = 1333
) -> List[Tuple[int, int, int, int]]:
    """
    Calculate window positions for left-to-right sliding on landscape images.

    Args:
        image_shape: (height, width) of original image
        num_windows: Number of windows to divide the image into
        target_short_side: GroundingDINO short side target (800)
        target_long_side: GroundingDINO long side max (1333)

    Returns:
        List of window coords: [(x1, y1, x2, y2), ...]
        where each window is (left, top, right, bottom) in pixels
    """
    height, width = image_shape[:2]

    # Calculate scale factor based on height (short side for landscape)
    scale_factor = target_short_side / height

    # Calculate window width so that after resize, width ≈ target_long_side
    window_width = int(target_long_side / scale_factor)

    # Window height is full image height
    window_height = height

    # If image is smaller than one window, just use the whole image
    if width <= window_width:
        return [(0, 0, width, height)]

    # Calculate stride to fit exactly num_windows
    # We want: (num_windows - 1) * stride + window_width = width
    # So: stride = (width - window_width) / (num_windows - 1)
    if num_windows == 1:
        stride = 0
    else:
        stride = int((width - window_width) / (num_windows - 1))

    # Generate window coordinates
    windows = []

    for i in range(num_windows):
        x_start = i * stride
        x_end = min(x_start + window_width, width)

        # Add window: (left, top, right, bottom)
        windows.append((x_start, 0, x_end, window_height))

        # If this window reaches the end, we're done
        if x_end >= width:
            break

    return windows


def detect_in_window(
        image: np.ndarray,
        window_coords: Tuple[int, int, int, int],
        model,
        text_prompt: str,
        box_threshold: float,
        text_threshold: float,
        device: str
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Extract window from image and run detection.

    Args:
        image: Full original image (numpy array, RGB format)
        window_coords: (x1, y1, x2, y2) of window in original image
        model: Loaded GroundingDINO model
        text_prompt: Detection prompt
        box_threshold: Detection threshold
        text_threshold: Text matching threshold
        device: 'cuda' or 'cpu'

    Returns:
        boxes: In window-relative normalized coordinates (cxcywh)
        logits: Confidence scores
        phrases: Detection labels
    """
    x1, y1, x2, y2 = window_coords

    # Extract window from original image
    window_image = image[y1:y2, x1:x2]

    # Convert to PIL for GroundingDINO transform
    window_pil = Image.fromarray(window_image)

    # Apply GroundingDINO transforms
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image_transformed, _ = transform(window_pil, None)

    # Run detection
    boxes, logits, phrases = predict(
        model=model,
        image=image_transformed,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device
    )

    return boxes, logits, phrases


def transform_boxes_to_original(
        boxes: torch.Tensor,
        window_coords: Tuple[int, int, int, int],
        window_shape: Tuple[int, int],
        original_shape: Tuple[int, int]
) -> torch.Tensor:
    """
    Transform boxes from window coordinates to original image coordinates.

    Args:
        boxes: Normalized boxes (cxcywh, 0-1) relative to window
        window_coords: (x1, y1, x2, y2) of window in original image
        window_shape: (height, width) of the window
        original_shape: (height, width) of original image

    Returns:
        boxes: Normalized boxes (cxcywh, 0-1) relative to original image
    """
    if boxes.nelement() == 0:
        return boxes

    x1, y1, x2, y2 = window_coords
    window_h, window_w = window_shape
    orig_h, orig_w = original_shape

    # Clone to avoid modifying original
    transformed_boxes = boxes.clone()

    # Boxes are in normalized cxcywh format relative to window
    # Convert to pixel coordinates in window
    cx = transformed_boxes[:, 0] * window_w
    cy = transformed_boxes[:, 1] * window_h
    w = transformed_boxes[:, 2] * window_w
    h = transformed_boxes[:, 3] * window_h

    # Offset by window position in original image
    cx = cx + x1
    cy = cy + y1

    # Normalize to original image dimensions
    transformed_boxes[:, 0] = cx / orig_w
    transformed_boxes[:, 1] = cy / orig_h
    transformed_boxes[:, 2] = w / orig_w
    transformed_boxes[:, 3] = h / orig_h

    return transformed_boxes


def calculate_iou(
        box1: torch.Tensor,
        box2: torch.Tensor,
        format: str = 'cxcywh'
) -> float:
    """
    Calculate Intersection over Union between two boxes.

    Args:
        box1, box2: Box tensors (normalized)
        format: 'cxcywh' or 'xyxy'

    Returns:
        IoU score (0-1)
    """
    # Convert to xyxy format if needed
    if format == 'cxcywh':
        box1_xyxy = box_ops.box_cxcywh_to_xyxy(box1.unsqueeze(0)).squeeze(0)
        box2_xyxy = box_ops.box_cxcywh_to_xyxy(box2.unsqueeze(0)).squeeze(0)
    else:
        box1_xyxy = box1
        box2_xyxy = box2

    # Calculate intersection
    x1_max = max(box1_xyxy[0].item(), box2_xyxy[0].item())
    y1_max = max(box1_xyxy[1].item(), box2_xyxy[1].item())
    x2_min = min(box1_xyxy[2].item(), box2_xyxy[2].item())
    y2_min = min(box1_xyxy[3].item(), box2_xyxy[3].item())

    # Check if boxes overlap
    if x2_min < x1_max or y2_min < y1_max:
        return 0.0

    intersection = (x2_min - x1_max) * (y2_min - y1_max)

    # Calculate union
    box1_area = (box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1])
    box2_area = (box2_xyxy[2] - box2_xyxy[0]) * (box2_xyxy[3] - box2_xyxy[1])
    union = box1_area + box2_area - intersection

    # Calculate IoU
    iou = intersection / (union + 1e-6)

    return iou.item() if isinstance(iou, torch.Tensor) else iou


def merge_overlapping_detections(
        all_boxes: List[torch.Tensor],
        all_logits: List[torch.Tensor],
        all_phrases: List[List[str]],
        iou_threshold: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Merge duplicate detections using NMS.

    Args:
        all_boxes: List of box tensors from each window (in original coords)
        all_logits: List of confidence scores from each window
        all_phrases: List of label lists from each window
        iou_threshold: IoU threshold for considering boxes duplicates

    Returns:
        merged_boxes: Deduplicated boxes
        merged_logits: Corresponding confidence scores
        merged_phrases: Corresponding labels
    """
    # Concatenate all detections
    if len(all_boxes) == 0 or all(b.nelement() == 0 for b in all_boxes):
        return torch.empty(0, 4), torch.empty(0), []

    # Filter out empty tensors and concatenate
    valid_boxes = [b for b in all_boxes if b.nelement() > 0]
    valid_logits = [l for l in all_logits if l.nelement() > 0]
    valid_phrases = [p for p in all_phrases if len(p) > 0]

    if len(valid_boxes) == 0:
        return torch.empty(0, 4), torch.empty(0), []

    boxes = torch.cat(valid_boxes, dim=0)
    logits = torch.cat(valid_logits, dim=0)
    phrases = []
    for phrase_list in valid_phrases:
        phrases.extend(phrase_list)

    # If only one detection, return as-is
    if boxes.shape[0] == 1:
        return boxes, logits, phrases

    # Convert to xyxy for NMS
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes)

    # Perform NMS using torchvision
    keep_indices = torchvision.ops.nms(
        boxes_xyxy,
        logits,
        iou_threshold=iou_threshold
    )

    # Filter results
    merged_boxes = boxes[keep_indices]
    merged_logits = logits[keep_indices]
    merged_phrases = [phrases[i] for i in keep_indices.cpu().numpy()]

    return merged_boxes, merged_logits, merged_phrases


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


def detect_insects_sliding_window(
        image_path: Union[str, np.ndarray],
        text_prompt: str = "insect",
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        num_windows: int = 3,
        nms_iou_threshold: float = 0.5,
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
    Detect insects using sliding window approach to preserve high-res details.

    Args:
        image_path: Path to image file or cv2 image array (BGR format)
        text_prompt: Text prompt for detection (default: "insect")
        box_threshold: Confidence threshold for keeping bounding boxes (0-1)
        text_threshold: Confidence threshold that boxes match the prompt (0-1)
        num_windows: Number of windows to divide the image into (default: 3)
        nms_iou_threshold: IoU threshold for merging duplicate detections (0-1)
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
        >>> boxes, scores, labels = detect_insects_sliding_window(  # doctest: +SKIP
        ...     "moth_sheet.jpg",
        ...     box_threshold=0.35,
        ...     num_windows=4,
        ...     visualize=True
        ... )
        >>> print(f"Detected {len(boxes)} insects")  # doctest: +SKIP
        Detected 42 insects
    """

    # Load image
    if verbose:
        print("Loading image...")

    if isinstance(image_path, str):
        image_source = Image.open(image_path).convert("RGB")
        image = np.asarray(image_source)
    elif isinstance(image_path, np.ndarray):
        # Assume BGR format from cv2, convert to RGB
        image = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
        image_source = Image.fromarray(image)
    else:
        raise TypeError("image_path must be a file path string or numpy array")

    original_shape = image.shape[:2]  # (height, width)

    if verbose:
        print(f"Image shape: {original_shape[0]}×{original_shape[1]}")

    # Calculate sliding windows
    if verbose:
        print(f"Calculating {num_windows} sliding windows...")

    windows = calculate_sliding_windows(
        image_shape=original_shape,
        num_windows=num_windows
    )

    if verbose:
        print(f"Generated {len(windows)} windows")
        for i, (x1, y1, x2, y2) in enumerate(windows):
            print(f"  Window {i + 1}: [{x1}:{x2}, {y1}:{y2}] ({x2 - x1}×{y2 - y1})")

    # Load model once
    if verbose:
        print("Loading GroundingDINO model...")

    model = _load_grounding_dino(
        repo_id=model_repo,
        filename=model_file,
        config_filename=config_file,
        use_gpu=use_gpu
    )

    device = 'cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu'

    # Process each window
    all_boxes = []
    all_logits = []
    all_phrases = []

    if verbose:
        print(f"\nDetecting '{text_prompt}' in each window...")

    for i, window_coords in enumerate(windows):
        if verbose:
            print(f"  Processing window {i + 1}/{len(windows)}...", end=" ")

        # Detect in window
        boxes, logits, phrases = detect_in_window(
            image=image,
            window_coords=window_coords,
            model=model,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device
        )

        num_detections = boxes.shape[0] if boxes.nelement() > 0 else 0
        if verbose:
            print(f"found {num_detections} insects")

        # Transform boxes to original image coordinates
        if num_detections > 0:
            x1, y1, x2, y2 = window_coords
            window_shape = (y2 - y1, x2 - x1)

            boxes_original = transform_boxes_to_original(
                boxes=boxes,
                window_coords=window_coords,
                window_shape=window_shape,
                original_shape=original_shape
            )

            all_boxes.append(boxes_original)
            all_logits.append(logits)
            all_phrases.append(phrases)

    # Merge overlapping detections
    if verbose:
        total_before_merge = sum(b.shape[0] for b in all_boxes if b.nelement() > 0)
        print(f"\nMerging overlapping detections (IoU threshold: {nms_iou_threshold})...")
        print(f"  Total detections before merge: {total_before_merge}")

    merged_boxes, merged_logits, merged_phrases = merge_overlapping_detections(
        all_boxes=all_boxes,
        all_logits=all_logits,
        all_phrases=all_phrases,
        iou_threshold=nms_iou_threshold
    )

    num_final = merged_boxes.shape[0] if merged_boxes.nelement() > 0 else 0
    if verbose:
        print(f"  Final detections after merge: {num_final}")
        if total_before_merge > 0:
            duplicates_removed = total_before_merge - num_final
            print(f"  Duplicates removed: {duplicates_removed}")

    # Visualize if requested
    annotated_image = None
    if visualize or save_visualization or return_image:
        if num_final > 0:
            # Convert merged boxes to format expected by annotate
            annotated_frame = annotate(
                image_source=image,
                boxes=merged_boxes,
                logits=merged_logits,
                phrases=merged_phrases
            )

            # annotate returns RGB already
            annotated_image = annotated_frame

            if visualize:
                img_cv = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

                # Display with window that can be resized
                window_name = f"Sliding Window Detection - {num_final} insects found"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow(window_name, img_cv)
                print("\nPress any key to close visualization window...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if save_visualization:
                img_cv = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_visualization, img_cv)
                if verbose:
                    print(f"Saved visualization to {save_visualization}")
        else:
            if verbose:
                print("No insects detected - skipping visualization")

    # Return results
    if return_image:
        return merged_boxes, merged_logits, merged_phrases, annotated_image
    else:
        return merged_boxes, merged_logits, merged_phrases


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


# Basic usage with sliding window
boxes, scores, labels = detect_insects_sliding_window(
    "D:/moth_test_imgs/ama_2025-08-11_22_30_02.jpg",
    box_threshold=0.1,
    text_threshold=0.25,
    num_windows=3,
    nms_iou_threshold=0.5,
    visualize=True,
    verbose=True
)

print(f"\n{'=' * 60}")
print(f"Detection Summary:")
print(f"{'=' * 60}")
print(f"Total insects detected: {len(boxes)}")
print(f"Confidence scores: {scores.cpu().numpy()}")
print(f"Labels: {labels}")

# Save visualization and get bounding box coordinates
print(f"\n{'=' * 60}")
print(f"Processing with visualization save...")
print(f"{'=' * 60}")