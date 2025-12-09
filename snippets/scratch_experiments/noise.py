"""
Background Generator for Camouflage Studies
============================================
Generates biologically realistic backgrounds with control over key parameters:
- Grain size (spatial frequency): How fine or coarse the pattern is
- Contrast: Difference between light and dark areas
- Complexity (alpha): How "natural" the texture looks (1/f^alpha)
- Edge density: How many boundaries/transitions exist

Perfect for testing disruptive camouflage patterns!
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import filters
import warnings

warnings.filterwarnings('ignore')


def generate_background(
        width=400,
        height=400,
        grain_size=20,  # pixels - larger = coarser pattern (LOW spatial freq)
        contrast=0.5,  # 0-1 - how different light/dark areas are
        complexity=1.0,  # alpha value: 1.0 = natural, 0 = random, 2 = smooth
        edge_density='auto',  # 'auto', 'low', 'medium', 'high', or number
        seed=None
):
    """
    Generate a biologically realistic background texture.

    Parameters
    ----------
    width, height : int
        Image dimensions in pixels

    grain_size : float
        Size of pattern elements in pixels. Think of this as:
        - 5-10: Fine texture (sand, fine ripples)
        - 20-40: Medium texture (bark, rocks)
        - 50-100: Coarse texture (large patches, clouds)

    contrast : float (0-1)
        How different light and dark areas are:
        - 0.2-0.3: Low contrast (overcast, deep shade)
        - 0.4-0.6: Medium contrast (typical lighting)
        - 0.7-0.9: High contrast (dappled sunlight)

    complexity : float
        Power law exponent (alpha) controlling naturalness:
        - 0.0: White noise (totally random, unnatural)
        - 1.0: Pink/1f noise (most natural - RECOMMENDED)
        - 1.5-2.0: Smoother, more blob-like
        - 0.5: Rougher, more chaotic

    edge_density : str or float
        Controls number of boundaries. Options:
        - 'auto': Determined by grain_size
        - 'low': Few boundaries (smooth gradients)
        - 'medium': Moderate boundaries
        - 'high': Many boundaries (busy pattern)
        - Or specify number of quantization levels (2-20)

    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    image : ndarray (height x width)
        Generated background image, values 0-1
    """

    if seed is not None:
        np.random.seed(seed)

    # Step 1: Create white noise
    white_noise = np.random.randn(height, width)

    # Step 2: Apply 1/f^alpha filter in frequency domain
    # This creates the natural texture structure
    fft_noise = np.fft.fft2(white_noise)

    # Create frequency coordinates
    kx = np.fft.fftfreq(width, d=1.0)
    ky = np.fft.fftfreq(height, d=1.0)
    KX, KY = np.meshgrid(kx, ky)

    # Radial frequency (distance from DC component)
    K = np.sqrt(KX ** 2 + KY ** 2)
    K[0, 0] = 1.0  # Avoid division by zero

    # Apply power law filter: 1/f^alpha
    power_filter = 1.0 / (K ** complexity)

    # Apply filter
    fft_filtered = fft_noise * power_filter

    # Transform back to spatial domain
    image = np.fft.ifft2(fft_filtered).real

    # Step 3: Apply grain size (smoothing)
    # Larger grain_size = smoother, larger features
    sigma = grain_size / 6.0  # Convert to Gaussian sigma
    image = ndimage.gaussian_filter(image, sigma=sigma)

    # Step 4: Normalize to 0-1 range
    image = (image - image.mean()) / (image.std() + 1e-10)

    # Step 5: Apply contrast
    # Center at 0.5, scale by contrast
    image = image * contrast + 0.5

    # Clip to valid range
    image = np.clip(image, 0, 1)

    # Step 6: Adjust edge density by quantization
    if edge_density != 'auto':
        if edge_density == 'low':
            n_levels = 3
        elif edge_density == 'medium':
            n_levels = 6
        elif edge_density == 'high':
            n_levels = 12
        else:
            n_levels = int(edge_density)

        # Quantize (posterize) to create edges
        image = np.round(image * (n_levels - 1)) / (n_levels - 1)

    return image


def measure_background_properties(image):
    """
    Measure key properties of a background image.

    Returns
    -------
    dict with:
        - grain_size: Approximate size of pattern elements
        - contrast: RMS contrast
        - complexity: Estimated alpha value
        - edge_density: Edges per 1000 pixels
    """
    # Contrast (RMS)
    contrast = np.std(image)

    # Edge density using Sobel filter
    edges = filters.sobel(image)
    edge_density = np.sum(edges > 0.1) / image.size * 1000

    # Estimate grain size from autocorrelation
    from scipy.signal import correlate2d
    center = image.shape[0] // 2
    autocorr = correlate2d(image, image, mode='same')
    autocorr_slice = autocorr[center, center:]

    # Find first point where autocorr drops below 0.5
    try:
        grain_size = np.where(autocorr_slice < autocorr_slice[0] * 0.5)[0][0] * 2
    except:
        grain_size = np.nan

    # Estimate complexity (alpha) from power spectrum
    fft_img = np.fft.fft2(image)
    power_spectrum = np.abs(np.fft.fftshift(fft_img)) ** 2

    # Radial average
    center_y, center_x = np.array(power_spectrum.shape) // 2
    y, x = np.ogrid[:power_spectrum.shape[0], :power_spectrum.shape[1]]
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2).astype(int)

    max_r = min(center_x, center_y)
    radial_mean = np.bincount(r.ravel(), power_spectrum.ravel()) / np.bincount(r.ravel())

    # Fit power law in log space (ignoring DC and very high freq)
    freqs = np.arange(1, min(50, len(radial_mean)))
    if len(freqs) > 10:
        log_freqs = np.log(freqs)
        log_power = np.log(radial_mean[freqs] + 1e-10)

        # Linear fit
        coeffs = np.polyfit(log_freqs, log_power, 1)
        complexity = -coeffs[0]  # Negative slope gives alpha
    else:
        complexity = np.nan

    return {
        'grain_size': grain_size,
        'contrast': contrast,
        'complexity': complexity,
        'edge_density': edge_density
    }


def visualize_parameter_effects():
    """
    Create a comprehensive visualization showing how each parameter affects the output.
    """
    fig = plt.figure(figsize=(16, 12))

    # Set up base parameters
    base_params = {
        'width': 300,
        'height': 300,
        'grain_size': 30,
        'contrast': 0.5,
        'complexity': 1.0,
        'edge_density': 'auto',
        'seed': 42
    }

    # 1. GRAIN SIZE variation
    grain_sizes = [10, 30, 60]
    for i, gs in enumerate(grain_sizes):
        params = base_params.copy()
        params['grain_size'] = gs
        img = generate_background(**params)

        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
        plt.title(f'Grain Size = {gs}\n({"Fine" if gs < 20 else "Medium" if gs < 50 else "Coarse"})')
        plt.axis('off')

        # Add measurement
        props = measure_background_properties(img)
        plt.text(10, 280, f'Measured: {props["grain_size"]:.0f}px',
                 color='red', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Add label for row
    plt.subplot(4, 4, 1)
    plt.text(-0.5, 0.5, 'GRAIN SIZE\n(spatial frequency)',
             transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
             verticalalignment='center', rotation=90)

    # 2. CONTRAST variation
    contrasts = [0.2, 0.5, 0.8]
    for i, c in enumerate(contrasts):
        params = base_params.copy()
        params['contrast'] = c
        img = generate_background(**params)

        ax = plt.subplot(4, 4, i + 5)
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
        plt.title(f'Contrast = {c:.1f}\n({"Low" if c < 0.4 else "Medium" if c < 0.7 else "High"})')
        plt.axis('off')

        props = measure_background_properties(img)
        plt.text(10, 280, f'Measured: {props["contrast"]:.2f}',
                 color='red', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.subplot(4, 4, 5)
    plt.text(-0.5, 0.5, 'CONTRAST\n(light/dark difference)',
             transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
             verticalalignment='center', rotation=90)

    # 3. COMPLEXITY (alpha) variation
    complexities = [0.0, 1.0, 2.0]
    complexity_labels = ['Random\n(white noise)', 'Natural\n(pink noise)', 'Smooth\n(brown noise)']
    for i, (comp, label) in enumerate(zip(complexities, complexity_labels)):
        params = base_params.copy()
        params['complexity'] = comp
        img = generate_background(**params)

        ax = plt.subplot(4, 4, i + 9)
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
        plt.title(f'Complexity (α) = {comp:.1f}\n{label}')
        plt.axis('off')

        props = measure_background_properties(img)
        plt.text(10, 280, f'Measured α: {props["complexity"]:.2f}',
                 color='red', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.subplot(4, 4, 9)
    plt.text(-0.5, 0.5, 'COMPLEXITY\n(naturalness)',
             transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
             verticalalignment='center', rotation=90)

    # 4. EDGE DENSITY variation
    edge_densities = ['low', 'medium', 'high']
    for i, ed in enumerate(edge_densities):
        params = base_params.copy()
        params['edge_density'] = ed
        img = generate_background(**params)

        ax = plt.subplot(4, 4, i + 13)
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
        plt.title(f'Edge Density = {ed.upper()}')
        plt.axis('off')

        props = measure_background_properties(img)
        plt.text(10, 280, f'Measured: {props["edge_density"]:.0f}/1k px',
                 color='red', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.subplot(4, 4, 13)
    plt.text(-0.5, 0.5, 'EDGE DENSITY\n(boundaries)',
             transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
             verticalalignment='center', rotation=90)

    # Add 4th example showing combined effect
    ax = plt.subplot(4, 4, 16)
    params = base_params.copy()
    params['grain_size'] = 25
    params['contrast'] = 0.6
    params['complexity'] = 1.0
    params['edge_density'] = 'medium'
    img = generate_background(**params)
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    plt.title('Typical Natural\nBackground\n(all parameters\nbalanced)')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('background_parameters.png', dpi=150, bbox_inches='tight')
    print("Saved: background_parameters.png")
    plt.show()


def compare_habitat_types():
    """
    Generate example backgrounds mimicking different habitat types.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Define habitat types with realistic parameters
    habitats = {
        'Still Water\n(calm pond)': {
            'grain_size': 50,
            'contrast': 0.3,
            'complexity': 1.5,
            'edge_density': 'low'
        },
        'Moving Water\n(stream/rapids)': {
            'grain_size': 15,
            'contrast': 0.6,
            'complexity': 1.0,
            'edge_density': 'high'
        },
        'Tree Bark\n(oak)': {
            'grain_size': 25,
            'contrast': 0.5,
            'complexity': 1.2,
            'edge_density': 'medium'
        },
        'Leaf Litter\n(forest floor)': {
            'grain_size': 20,
            'contrast': 0.4,
            'complexity': 0.8,
            'edge_density': 'high'
        },
        'Rocky Surface\n(stones)': {
            'grain_size': 35,
            'contrast': 0.45,
            'complexity': 1.0,
            'edge_density': 'medium'
        },
        'Grass/Reeds\n(vertical)': {
            'grain_size': 12,
            'contrast': 0.55,
            'complexity': 0.9,
            'edge_density': 'high'
        }
    }

    for ax, (name, params) in zip(axes.flat, habitats.items()):
        params['width'] = 300
        params['height'] = 300
        params['seed'] = 42

        img = generate_background(**params)

        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.axis('off')

        # Add parameter info
        info_text = (f"Grain: {params['grain_size']}\n"
                     f"Contrast: {params['contrast']}\n"
                     f"α: {params['complexity']}\n"
                     f"Edges: {params['edge_density']}")
        ax.text(10, 280, info_text, fontsize=7,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))

    plt.tight_layout()
    plt.savefig('habitat_examples.png', dpi=150, bbox_inches='tight')
    print("Saved: habitat_examples.png")
    plt.show()


def demonstrate_dragonfly_camouflage():
    """
    Show how dragonfly patterns might work on different backgrounds.
    """
    fig = plt.figure(figsize=(14, 8))

    # Generate two habitat types
    still_water = generate_background(
        width=400, height=400,
        grain_size=50, contrast=0.3, complexity=1.5, edge_density='low', seed=42
    )

    moving_water = generate_background(
        width=400, height=400,
        grain_size=15, contrast=0.6, complexity=1.0, edge_density='high', seed=42
    )

    # Create simple dragonfly silhouettes
    def create_dragonfly(pattern_type, width=60, height=20):
        """Create a simple dragonfly shape with different patterns."""
        dragonfly = np.ones((height, width)) * 0.5

        # Body (simple rectangle)
        body_width = 10
        body_start = (width - body_width) // 2
        dragonfly[:, body_start:body_start + body_width] = 0.3

        # Wings (triangles on sides)
        for i in range(height):
            wing_width = int((height - abs(i - height // 2)) * 1.5)
            # Left wing
            dragonfly[i, max(0, body_start - wing_width):body_start] = 0.7
            # Right wing
            dragonfly[i, body_start + body_width:min(width, body_start + body_width + wing_width)] = 0.7

        if pattern_type == 'striped':
            # Add yellow-black stripes
            stripe_width = 4
            for i in range(0, width, stripe_width * 2):
                dragonfly[:, i:i + stripe_width] = 0.9  # Yellow (light)
                dragonfly[:, i + stripe_width:i + stripe_width * 2] = 0.1  # Black (dark)
        elif pattern_type == 'uniform_light':
            dragonfly[:, :] = 0.7
        elif pattern_type == 'uniform_dark':
            dragonfly[:, :] = 0.3

        return dragonfly

    # Test combinations
    patterns = ['striped', 'uniform_light', 'uniform_dark']
    pattern_names = ['Striped\n(Yellow-Black)', 'Uniform\n(Light)', 'Uniform\n(Dark)']

    for col, (pattern, pname) in enumerate(zip(patterns, pattern_names)):
        dragonfly = create_dragonfly(pattern)

        # On still water
        ax1 = plt.subplot(2, 3, col + 1)
        bg_with_df = still_water.copy()
        y_pos, x_pos = 190, 170
        bg_with_df[y_pos:y_pos + dragonfly.shape[0], x_pos:x_pos + dragonfly.shape[1]] = dragonfly

        plt.imshow(bg_with_df, cmap='gray', vmin=0, vmax=1)
        if col == 0:
            plt.ylabel('STILL WATER\n(calm pond)', fontsize=11, fontweight='bold')
        plt.title(pname, fontsize=10)
        plt.axis('off')

        # On moving water
        ax2 = plt.subplot(2, 3, col + 4)
        bg_with_df = moving_water.copy()
        bg_with_df[y_pos:y_pos + dragonfly.shape[0], x_pos:x_pos + dragonfly.shape[1]] = dragonfly

        plt.imshow(bg_with_df, cmap='gray', vmin=0, vmax=1)
        if col == 0:
            plt.ylabel('MOVING WATER\n(stream/rapids)', fontsize=11, fontweight='bold')
        plt.axis('off')

    plt.suptitle('Dragonfly Camouflage Test: Which Pattern Works Best on Which Background?',
                 fontsize=13, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig('dragonfly_camouflage_demo.png', dpi=150, bbox_inches='tight')
    print("Saved: dragonfly_camouflage_demo.png")
    plt.show()


def create_experimental_stimuli(n_backgrounds=5):
    """
    Generate a set of backgrounds for an actual experiment.
    Saves individual files that can be used in your camouflage study.
    """
    import os

    # Create output directory
    os.makedirs('experiment_backgrounds', exist_ok=True)

    # Define parameter ranges for factorial design
    grain_sizes = [15, 30, 50]  # fine, medium, coarse
    contrasts = [0.3, 0.5, 0.7]  # low, medium, high

    print("Generating experimental backgrounds...")
    print("=" * 50)

    bg_index = 0
    for grain in grain_sizes:
        for contrast in contrasts:
            for rep in range(n_backgrounds):
                # Generate background
                bg = generate_background(
                    width=600,
                    height=600,
                    grain_size=grain,
                    contrast=contrast,
                    complexity=1.0,  # Keep natural
                    edge_density='auto',
                    seed=bg_index
                )

                # Measure properties
                props = measure_background_properties(bg)

                # Save
                filename = f'bg_grain{grain}_contrast{int(contrast * 10)}_rep{rep}.png'
                filepath = os.path.join('experiment_backgrounds', filename)

                plt.imsave(filepath, bg, cmap='gray', vmin=0, vmax=1)

                print(f"Saved: {filename}")
                print(f"  Measured - Grain: {props['grain_size']:.1f}, "
                      f"Contrast: {props['contrast']:.2f}, "
                      f"Alpha: {props['complexity']:.2f}, "
                      f"Edges: {props['edge_density']:.0f}/1k")

                bg_index += 1

    print("=" * 50)
    print(f"Generated {bg_index} backgrounds in 'experiment_backgrounds/' folder")
    print(f"Factorial design: {len(grain_sizes)} grain sizes × {len(contrasts)} contrasts × {n_backgrounds} reps")


if __name__ == '__main__':
    print("=" * 60)
    print("BACKGROUND GENERATOR FOR CAMOUFLAGE STUDIES")
    print("=" * 60)
    print()

    # 1. Show how parameters work
    print("1. Visualizing parameter effects...")
    visualize_parameter_effects()
    print()

    # 2. Show habitat examples
    print("2. Generating habitat type examples...")
    compare_habitat_types()
    print()

    # 3. Show dragonfly camouflage demo
    print("3. Creating dragonfly camouflage demonstration...")
    demonstrate_dragonfly_camouflage()
    print()

    # 4. Option to generate experimental stimuli
    print("4. Generate experimental backgrounds? (y/n)")
    response = input("> ").strip().lower()
    if response == 'y':
        n_reps = int(input("How many repetitions per condition? (recommended: 3-5): "))
        create_experimental_stimuli(n_backgrounds=n_reps)

    print()
    print("=" * 60)
    print("DONE! Check the generated PNG files.")
    print("=" * 60)