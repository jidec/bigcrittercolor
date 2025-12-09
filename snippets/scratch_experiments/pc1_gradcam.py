"""
Multiple methods to visualize PC attention for BioEncoder
Includes: Grad-CAM, Integrated Gradients, Occlusion, Feature Visualization, and Activation-PC Correlation
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cv2
import os
from tqdm import tqdm

try:
    from bioencoder import utils
except ImportError:
    print("Warning: bioencoder not found.")


# =============================================================================
# Model Loading (same as before)
# =============================================================================

def load_bioencoder_model(config_path, swa_weights_path, stage="first"):
    """Load BioEncoder model from SWA checkpoint"""
    hyperparams = utils.load_yaml(config_path)
    backbone = hyperparams["model"]["backbone"]

    if stage == "second":
        num_classes = hyperparams["model"]["num_classes"]
    else:
        num_classes = None

    model = utils.build_model(
        backbone,
        second_stage=(stage == "second"),
        num_classes=num_classes,
        ckpt_pretrained=None,
    )

    checkpoint = torch.load(swa_weights_path, map_location='cpu')
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    print(f"Loaded model with backbone: {backbone}")
    return model, hyperparams


def get_bioencoder_target_layer(model, backbone):
    """Get the appropriate target layer"""
    if 'efficientnet' in backbone.lower():
        try:
            target_layer = model.encoder[2][6][-1]
            print(f"Using EfficientNet target layer: encoder[2][6][-1]")
        except (IndexError, AttributeError):
            target_layer = model.encoder[3]
            print(f"Using fallback target layer: encoder[3]")
    elif 'resnet' in backbone.lower():
        try:
            target_layer = model.encoder[2][-1][-1]
            print(f"Using ResNet target layer: encoder[2][-1][-1]")
        except (IndexError, AttributeError):
            target_layer = model.encoder[3]
            print(f"Using fallback target layer: encoder[3]")
    else:
        try:
            target_layer = model.encoder[2][-1][-1]
            print(f"Using target layer: encoder[2][-1][-1]")
        except (IndexError, AttributeError):
            target_layer = model.encoder[3]
            print(f"Using fallback target layer: encoder[3]")

    return target_layer


# =============================================================================
# METHOD 1: Grad-CAM (Original)
# =============================================================================

class GradCAMPC:
    """Compute Grad-CAM attention maps"""

    def __init__(self, model, target_layer, pc_loadings, device='cuda'):
        self.model = model
        self.target_layer = target_layer
        self.pc_loadings = torch.tensor(pc_loadings, dtype=torch.float32).to(device)
        self.device = device

        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, image_tensor):
        """Generate Grad-CAM"""
        image_tensor = image_tensor.to(self.device)
        image_tensor.requires_grad = True

        self.model.eval()
        with torch.set_grad_enabled(True):
            features = self.model.encoder(image_tensor)
            embedding = features.view(image_tensor.size(0), -1)

            if len(embedding.shape) == 1:
                embedding = embedding.unsqueeze(0)

            pc_score = torch.dot(embedding.squeeze(), self.pc_loadings)

            self.model.zero_grad()
            pc_score.backward()

        gradients = self.gradients
        activations = self.activations

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1).squeeze()

        cam = torch.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy(), pc_score.item(), embedding.detach().cpu().numpy()


# =============================================================================
# METHOD 2: Integrated Gradients
# =============================================================================

def integrated_gradients_pc(model, image, pc_loadings, device='cuda', baseline=None, steps=50):
    """
    Compute Integrated Gradients for PC
    More accurate attribution than Grad-CAM

    Args:
        model: BioEncoder model
        image: Input tensor [1, C, H, W]
        pc_loadings: PC loadings vector
        baseline: Baseline image (default: black image)
        steps: Number of interpolation steps

    Returns:
        attribution_map: [H, W] importance map
        pc_score: PC score for the image
    """
    if baseline is None:
        baseline = torch.zeros_like(image)

    image = image.to(device)
    baseline = baseline.to(device)
    pc_loadings = torch.tensor(pc_loadings, dtype=torch.float32).to(device)

    # Generate interpolated images
    alphas = torch.linspace(0, 1, steps, device=device)
    interpolated_images = baseline.unsqueeze(0) + alphas.view(-1, 1, 1, 1, 1) * (image - baseline).unsqueeze(0)

    gradients = []

    model.eval()
    for i in range(steps):
        img = interpolated_images[i].requires_grad_(True)

        # Forward pass
        embedding = model.encoder(img).view(1, -1)
        pc_score = torch.dot(embedding.squeeze(), pc_loadings)

        # Backward
        model.zero_grad()
        pc_score.backward()

        gradients.append(img.grad.detach())

    # Average gradients and multiply by difference
    avg_gradients = torch.stack(gradients).mean(dim=0)
    integrated_grads = (image - baseline) * avg_gradients

    # Sum across color channels and take absolute value
    attribution_map = integrated_grads.squeeze().abs().sum(dim=0).cpu().numpy()

    # Normalize
    attribution_map = attribution_map - attribution_map.min()
    if attribution_map.max() > 0:
        attribution_map = attribution_map / attribution_map.max()

    # Get final PC score
    with torch.no_grad():
        embedding = model.encoder(image).view(1, -1)
        pc_score = torch.dot(embedding.squeeze(), pc_loadings).item()

    return attribution_map, pc_score


# =============================================================================
# METHOD 3: Occlusion Analysis
# =============================================================================

def occlusion_pc(model, image, pc_loadings, device='cuda', patch_size=32, stride=16):
    """
    Occlusion-based importance map
    Shows which regions actually affect PC score

    Args:
        model: BioEncoder model
        image: Input tensor [1, C, H, W]
        pc_loadings: PC loadings vector
        patch_size: Size of occlusion patch
        stride: Stride for sliding window

    Returns:
        importance_map: [H, W] map showing importance of each region
        pc_orig: Original PC score
    """
    image = image.to(device)
    pc_loadings = torch.tensor(pc_loadings, dtype=torch.float32).to(device)

    _, _, H, W = image.shape

    # Get baseline PC score
    model.eval()
    with torch.no_grad():
        embedding_orig = model.encoder(image).view(1, -1)
        pc_orig = torch.dot(embedding_orig.squeeze(), pc_loadings).item()

    # Create importance map
    h_steps = (H - patch_size) // stride + 1
    w_steps = (W - patch_size) // stride + 1
    importance_map = np.zeros((h_steps, w_steps))

    print(f"      Running occlusion analysis ({h_steps}x{w_steps} patches)...")

    for i in range(h_steps):
        for j in range(w_steps):
            y = i * stride
            x = j * stride

            # Create occluded image
            img_occluded = image.clone()
            img_occluded[:, :, y:y+patch_size, x:x+patch_size] = 0  # Black out patch

            # Measure PC with occlusion
            with torch.no_grad():
                embedding_occ = model.encoder(img_occluded).view(1, -1)
                pc_occ = torch.dot(embedding_occ.squeeze(), pc_loadings).item()

            # Importance = how much PC changed
            importance_map[i, j] = abs(pc_orig - pc_occ)

    # Upsample to image size
    importance_map = cv2.resize(importance_map, (W, H), interpolation=cv2.INTER_LINEAR)

    # Normalize
    importance_map = importance_map - importance_map.min()
    if importance_map.max() > 0:
        importance_map = importance_map / importance_map.max()

    return importance_map, pc_orig


# =============================================================================
# METHOD 4: Feature Visualization (Generate Extremes)
# =============================================================================

def generate_pc_extremes(model, pc_loadings, device='cuda', steps=1000, lr=0.02,
                         image_size=224, l2_weight=0.01, tv_weight=0.01, blur_every=4):
    """
    Generate synthetic images that maximize/minimize PC
    Shows what model thinks are extreme cases

    Args:
        model: BioEncoder model
        pc_loadings: PC loadings vector
        steps: Optimization steps (increased for better results)
        lr: Learning rate (reduced for stability)
        image_size: Size of generated images
        l2_weight: L2 regularization weight
        tv_weight: Total variation weight (smoothness)
        blur_every: Apply Gaussian blur every N steps

    Returns:
        image_high: Image maximizing PC (e.g., uniform)
        image_low: Image minimizing PC (e.g., striped)
        scores: Dict with PC scores over time
    """
    pc_loadings = torch.tensor(pc_loadings, dtype=torch.float32).to(device)

    # Start with gray noise (closer to natural images)
    image_high = torch.ones(1, 3, image_size, image_size, device=device) * 0.5
    image_high += torch.randn_like(image_high) * 0.05
    image_low = torch.ones(1, 3, image_size, image_size, device=device) * 0.5
    image_low += torch.randn_like(image_low) * 0.05

    # Make them leaf tensors that require gradients
    image_high = image_high.detach().requires_grad_(True)
    image_low = image_low.detach().requires_grad_(True)

    optimizer_high = torch.optim.Adam([image_high], lr=lr)
    optimizer_low = torch.optim.Adam([image_low], lr=lr)

    scores_high = []
    scores_low = []

    print(f"      Generating PC extremes ({steps} steps)...")

    # Gaussian blur for smoothness
    from torch.nn.functional import conv2d
    kernel_size = 5
    sigma = 1.0
    kernel = torch.zeros(kernel_size, kernel_size, device=device)
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - kernel_size // 2, j - kernel_size // 2
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)

    def total_variation(img):
        """Compute total variation for smoothness"""
        tv_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).sum()
        tv_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).sum()
        return tv_h + tv_w

    model.eval()
    for i in range(steps):
        # Maximize PC
        embedding_high = model.encoder(image_high).view(1, -1)
        pc_high = torch.dot(embedding_high.squeeze(), pc_loadings)

        # Loss with multiple regularizations
        loss_high = (-pc_high +  # Maximize PC
                    l2_weight * torch.norm(image_high - 0.5) +  # Stay close to gray
                    tv_weight * total_variation(image_high))  # Smoothness

        optimizer_high.zero_grad()
        loss_high.backward()
        optimizer_high.step()

        # Minimize PC
        embedding_low = model.encoder(image_low).view(1, -1)
        pc_low = torch.dot(embedding_low.squeeze(), pc_loadings)

        loss_low = (pc_low +  # Minimize PC
                   l2_weight * torch.norm(image_low - 0.5) +
                   tv_weight * total_variation(image_low))

        optimizer_low.zero_grad()
        loss_low.backward()
        optimizer_low.step()

        # Apply blur periodically for smoothness
        if i % blur_every == 0:
            with torch.no_grad():
                padding = kernel_size // 2
                image_high.data = conv2d(image_high.data, kernel, padding=padding, groups=3)
                image_low.data = conv2d(image_low.data, kernel, padding=padding, groups=3)

        # Clip to valid range
        with torch.no_grad():
            image_high.clamp_(0, 1)
            image_low.clamp_(0, 1)

        if i % 100 == 0:
            scores_high.append(pc_high.item())
            scores_low.append(pc_low.item())
            if i % 200 == 0:
                print(f"         Step {i}: High={pc_high.item():.3f}, Low={pc_low.item():.3f}")

    return (image_high.detach().cpu(),
            image_low.detach().cpu(),
            {'high': scores_high, 'low': scores_low})


# =============================================================================
# METHOD 5: Activation-PC Correlation
# =============================================================================

def activation_pc_correlation(model, image_dataset, pc_scores, target_layer, device='cuda'):
    """
    Correlate activation patterns with PC scores across multiple images
    Shows which spatial locations consistently correlate with PC

    Args:
        model: BioEncoder model
        image_dataset: List of image tensors
        pc_scores: Corresponding PC scores for images
        target_layer: Layer to extract activations from

    Returns:
        correlation_map: [H, W] map of correlations
    """
    activations = []

    print(f"      Computing activations for {len(image_dataset)} images...")

    # Hook to capture activations
    captured_acts = []
    def hook_fn(module, input, output):
        captured_acts.append(output.detach())

    handle = target_layer.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        for image in image_dataset:
            image = image.to(device)
            _ = model.encoder(image)

            # Get activation and average across channels
            act = captured_acts[-1]
            act_spatial = act.mean(dim=1).flatten()  # [H*W]
            activations.append(act_spatial.cpu().numpy())

    handle.remove()

    activations = np.stack(activations)  # [N, H*W]
    pc_scores = np.array(pc_scores)

    print(f"      Computing correlations...")

    # Correlate each spatial location with PC scores
    correlations = np.zeros(activations.shape[1])
    for i in range(activations.shape[1]):
        correlations[i] = np.corrcoef(activations[:, i], pc_scores)[0, 1]

    # Reshape to spatial grid
    H = W = int(np.sqrt(activations.shape[1]))
    correlation_map = correlations.reshape(H, W)

    # Take absolute value and normalize
    correlation_map = np.abs(correlation_map)
    correlation_map = correlation_map - correlation_map.min()
    if correlation_map.max() > 0:
        correlation_map = correlation_map / correlation_map.max()

    return correlation_map


# =============================================================================
# Visualization Functions
# =============================================================================

def visualize_all_methods(image_path, maps_dict, pc_score, save_path=None, alpha=0.4):
    """
    Visualize all attribution methods side by side

    Args:
        image_path: Path to original image
        maps_dict: Dictionary with method names and attribution maps
        pc_score: PC score
        save_path: Where to save
    """
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)

    n_methods = len(maps_dict)
    fig, axes = plt.subplots(2, n_methods + 1, figsize=(5*(n_methods+1), 10))

    # Original image in first column
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title('Original Image', fontsize=12)
    axes[0, 0].axis('off')

    axes[1, 0].imshow(image_np)
    axes[1, 0].set_title(f'PC Score: {pc_score:.3f}', fontsize=12)
    axes[1, 0].axis('off')

    # Each method in subsequent columns
    for idx, (method_name, attribution_map) in enumerate(maps_dict.items(), start=1):
        # Resize to match image
        map_resized = cv2.resize(attribution_map, (image_np.shape[1], image_np.shape[0]))

        # Heatmap only (top row)
        im = axes[0, idx].imshow(map_resized, cmap='jet')
        axes[0, idx].set_title(method_name, fontsize=12)
        axes[0, idx].axis('off')
        plt.colorbar(im, ax=axes[0, idx], fraction=0.046)

        # Overlay (bottom row)
        heatmap = cv2.applyColorMap(np.uint8(255 * map_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = heatmap * alpha + image_np * (1 - alpha)
        overlay = overlay.astype(np.uint8)

        axes[1, idx].imshow(overlay)
        axes[1, idx].set_title(f'{method_name} Overlay', fontsize=12)
        axes[1, idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"      Saved comparison to {save_path}")

    plt.close('all')


def visualize_feature_extremes(image_high, image_low, scores, pc_component, save_path=None):
    """Visualize generated extreme images"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # High PC image
    img_high = image_high.squeeze().permute(1, 2, 0).numpy()
    axes[0].imshow(np.clip(img_high, 0, 1))
    axes[0].set_title(f'High PC{pc_component}\n(Uniform Pattern)', fontsize=14)
    axes[0].axis('off')

    # Low PC image
    img_low = image_low.squeeze().permute(1, 2, 0).numpy()
    axes[1].imshow(np.clip(img_low, 0, 1))
    axes[1].set_title(f'Low PC{pc_component}\n(Striped Pattern)', fontsize=14)
    axes[1].axis('off')

    # Score evolution
    axes[2].plot(scores['high'], label='High PC (Uniform)', linewidth=2)
    axes[2].plot(scores['low'], label='Low PC (Striped)', linewidth=2)
    axes[2].set_xlabel('Optimization Step (×100)', fontsize=12)
    axes[2].set_ylabel(f'PC{pc_component} Score', fontsize=12)
    axes[2].set_title('Score Evolution', fontsize=14)
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"      Saved extremes to {save_path}")

    plt.close('all')


# =============================================================================
# Main Pipeline
# =============================================================================

def compute_pc_visualizations(
    config_path,
    swa_weights_path,
    features_csv_path,
    image_paths,
    output_dir='pc_visualizations',
    stage='first',
    pc_component=1,
    methods=['gradcam', 'integrated_gradients', 'occlusion'],
    generate_extremes=False,
    correlation_images=None  # List of image paths for correlation analysis
):
    """
    Complete pipeline with multiple visualization methods

    Args:
        methods: List from ['gradcam', 'integrated_gradients', 'occlusion', 'correlation']
        generate_extremes: Whether to generate synthetic extreme images
        correlation_images: If provided, compute activation-PC correlation using these images
    """

    if isinstance(image_paths, str):
        image_paths = [image_paths]

    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print(f"PC{pc_component} Visualization Pipeline")
    print(f"Methods: {', '.join(methods)}")
    print("="*80)

    # Load model
    print("\n[1/6] Loading model...")
    model, hyperparams = load_bioencoder_model(config_path, swa_weights_path, stage)
    backbone = hyperparams["model"]["backbone"]
    target_layer = get_bioencoder_target_layer(model, backbone)

    # Compute PCA
    print("\n[2/6] Computing PCA...")
    features_df = pd.read_csv(features_csv_path)

    potential_cols = [col for col in features_df.columns
                     if col.startswith(('feature_', 'emb_', 'feat_'))]
    if not potential_cols:
        potential_cols = [col for col in features_df.columns if not col.startswith(('image'))]
        potential_cols = features_df[potential_cols].select_dtypes(include=[np.number]).columns.tolist()

    X = features_df[potential_cols].values
    X = np.where(np.isinf(X), np.nan, X)
    mask = ~np.isnan(X).any(axis=1)
    X_clean = X[mask]

    pca = PCA(n_components=10)
    pca.fit(X_clean)

    pc_idx = pc_component - 1
    pc_loadings = pca.components_[pc_idx]

    print(f"   PC{pc_component} explains {pca.explained_variance_ratio_[pc_idx]*100:.2f}% of variance")

    # Setup methods
    print("\n[3/6] Setting up visualization methods...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    gradcam = None
    if 'gradcam' in methods:
        gradcam = GradCAMPC(model, target_layer, pc_loadings, device=device)

    # Build transforms
    transforms = utils.build_transforms(hyperparams)
    if 'valid_transforms' in transforms:
        preprocess = transforms['valid_transforms']
    elif 'train_transforms' in transforms:
        preprocess = transforms['train_transforms']
    else:
        preprocess = list(transforms.values())[0]

    # Generate feature extremes if requested
    if generate_extremes:
        print("\n[4/6] Generating synthetic extreme images...")
        img_high, img_low, scores = generate_pc_extremes(
            model, pc_loadings, device=device, steps=1000, lr=0.02
        )
        save_path = os.path.join(output_dir, f'pc{pc_component}_extremes.png')
        visualize_feature_extremes(img_high, img_low, scores, pc_component, save_path)

    # Compute activation correlation if dataset provided
    correlation_map = None
    if 'correlation' in methods and correlation_images is not None:
        print("\n[5/6] Computing activation-PC correlation...")

        # Load images and compute PC scores
        correlation_tensors = []
        correlation_pc_scores = []

        for img_path in correlation_images:
            image = Image.open(img_path).convert('RGB')
            image_np = np.array(image)
            transformed = preprocess(image=image_np)
            transformed_image = transformed['image']

            if isinstance(transformed_image, torch.Tensor):
                image_tensor = transformed_image.unsqueeze(0)
            else:
                image_tensor = torch.from_numpy(transformed_image).permute(2, 0, 1).unsqueeze(0).float()

            if image_tensor.max() > 1.0:
                image_tensor = image_tensor / 255.0

            correlation_tensors.append(image_tensor)

            # Get PC score
            with torch.no_grad():
                image_tensor = image_tensor.to(device)
                embedding = model.encoder(image_tensor).view(1, -1)
                pc_score = torch.dot(embedding.squeeze(), torch.tensor(pc_loadings, device=device)).item()
                correlation_pc_scores.append(pc_score)

        correlation_map = activation_pc_correlation(
            model, correlation_tensors, correlation_pc_scores, target_layer, device
        )

    # Process images
    print(f"\n[6/6] Processing {len(image_paths)} images...")

    all_results = []

    for idx, image_path in enumerate(image_paths):
        print(f"\n   [{idx+1}/{len(image_paths)}] {os.path.basename(image_path)}")

        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image)
            transformed = preprocess(image=image_np)
            transformed_image = transformed['image']

            if isinstance(transformed_image, torch.Tensor):
                image_tensor = transformed_image.unsqueeze(0)
            else:
                image_tensor = torch.from_numpy(transformed_image).permute(2, 0, 1).unsqueeze(0).float()

            if image_tensor.max() > 1.0:
                image_tensor = image_tensor / 255.0

            # Compute all requested methods
            maps_dict = {}
            pc_score = None

            if 'gradcam' in methods:
                print("      Computing Grad-CAM...")
                cam, pc_score, _ = gradcam.generate_cam(image_tensor)
                maps_dict['Grad-CAM'] = cam

            if 'integrated_gradients' in methods:
                print("      Computing Integrated Gradients...")
                ig_map, pc_score = integrated_gradients_pc(
                    model, image_tensor, pc_loadings, device=device, steps=50
                )
                maps_dict['Integrated Gradients'] = ig_map

            if 'occlusion' in methods:
                print("      Computing Occlusion...")
                occ_map, pc_score = occlusion_pc(
                    model, image_tensor, pc_loadings, device=device,
                    patch_size=32, stride=16
                )
                maps_dict['Occlusion'] = occ_map

            if correlation_map is not None:
                maps_dict['Activation Correlation'] = correlation_map

            # Visualize all methods
            basename = os.path.splitext(os.path.basename(image_path))[0]
            save_path = os.path.join(output_dir, f'{basename}_pc{pc_component}_all_methods.png')
            visualize_all_methods(image_path, maps_dict, pc_score, save_path)

            all_results.append({
                'image_path': image_path,
                'basename': basename,
                'pc_score': pc_score,
                'maps': maps_dict
            })

        except Exception as e:
            print(f"      ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print(f"Done! Processed {len(all_results)} images")
    print(f"Results saved to: {output_dir}")
    print("="*80)

    return all_results


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":

    CONFIG_PATH = "D:/bcc/bioencoder_training/bioencoder_configs/swa_stage1.yml"
    SWA_WEIGHTS = "D:/bcc/bioencoder_training/bioencoder_wd/weights/swa.pt"
    FEATURES_CSV = "D:/GitProjects/dragonfly-color-macro-analysis/data/bioencodings.csv"
    
    # Images to analyze
    IMAGE_PATHS = [
        "D:/bcc/new_random_dragonflies2/masks/INATRANDOM-116732733_mask.png",
        "D:/bcc/new_random_dragonflies2/masks/INATRANDOM-144923_mask.png",
        "D:/bcc/new_random_dragonflies2/masks/INATRANDOM-112457408_mask.png",
    ]

    # For correlation analysis, provide a larger sample of images
    # import glob
    # CORRELATION_IMAGES = glob.glob("D:/bcc/new_random_dragonflies/masks/*.png")[:50]
    CORRELATION_IMAGES = None

    results = compute_pc_visualizations(
        config_path=CONFIG_PATH,
        swa_weights_path=SWA_WEIGHTS,
        features_csv_path=FEATURES_CSV,
        image_paths=IMAGE_PATHS,
        output_dir='pc1_all_methods',
        pc_component=1,
        methods=['gradcam', 'integrated_gradients', 'occlusion'],  # Choose methods
        generate_extremes=True,  # Generate synthetic extremes
        correlation_images=CORRELATION_IMAGES  # Provide for correlation analysis
    )