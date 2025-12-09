"""
Interactive Camouflage Detection Experiment
===========================================
Human trials to test dragonfly detection on various background types.
Records reaction time, click accuracy, and all background parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
from scipy import ndimage
from PIL import Image
import time
import os
import csv
from datetime import datetime
import glob


def generate_background(width=800, height=800, grain_size=20, contrast=0.5,
                        complexity=1.0, edge_density='auto', seed=None):
    """Generate biologically realistic background texture."""
    if seed is not None:
        np.random.seed(seed)

    # Create white noise
    white_noise = np.random.randn(height, width)

    # Apply 1/f^alpha filter
    fft_noise = np.fft.fft2(white_noise)
    kx = np.fft.fftfreq(width, d=1.0)
    ky = np.fft.fftfreq(height, d=1.0)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)
    K[0, 0] = 1.0

    power_filter = 1.0 / (K ** complexity)
    fft_filtered = fft_noise * power_filter
    image = np.fft.ifft2(fft_filtered).real

    # Apply grain size smoothing
    sigma = grain_size / 6.0
    image = ndimage.gaussian_filter(image, sigma=sigma)

    # Normalize and apply contrast
    image = (image - image.mean()) / (image.std() + 1e-10)
    image = image * contrast + 0.5
    image = np.clip(image, 0, 1)

    # Apply edge density
    if edge_density != 'auto':
        if edge_density == 'low':
            n_levels = 3
        elif edge_density == 'medium':
            n_levels = 6
        elif edge_density == 'high':
            n_levels = 12
        else:
            n_levels = int(edge_density)
        image = np.round(image * (n_levels - 1)) / (n_levels - 1)

    return image


class CamouflageExperiment:
    """Interactive camouflage detection experiment."""

    def __init__(self,
                 user_name,
                 dragonfly_folder,
                 n_trials=50,
                 image_size=(800, 800),
                 output_file='camouflage_data.csv',
                 # Background parameter ranges
                 grain_size_range=(10, 60),
                 contrast_range=(0.2, 0.8),
                 complexity_range=(0.5, 1.5),
                 edge_density_options=['auto', 'low', 'medium', 'high'],
                 # Dragonfly parameters
                 dragonfly_scale_range=(0.8, 1.5),
                 min_border_distance=100,
                 # Display parameters
                 fixation_duration=0.5,
                 max_trial_time=30.0,
                 show_feedback=True,
                 debug_mode=False):
        """
        Initialize the camouflage detection experiment.

        Parameters
        ----------
        user_name : str
            Identifier for the participant (for mixed-effects models)
        dragonfly_folder : str
            Path to folder containing dragonfly PNG images with transparency
        n_trials : int
            Number of trials to run
        image_size : tuple
            (width, height) of the display in pixels
        output_file : str
            CSV file to save trial data
        grain_size_range : tuple
            (min, max) grain size in pixels (spatial frequency)
        contrast_range : tuple
            (min, max) contrast values (0-1)
        complexity_range : tuple
            (min, max) alpha values for 1/f^alpha noise
        edge_density_options : list
            Options for edge density ('auto', 'low', 'medium', 'high', or numbers)
        dragonfly_scale_range : tuple
            (min, max) scale factors for dragonfly size
        min_border_distance : int
            Minimum distance from edge to place dragonfly (pixels)
        fixation_duration : float
            Duration to show fixation cross before trial (seconds)
        max_trial_time : float
            Maximum time allowed per trial (seconds)
        show_feedback : bool
            Whether to show feedback after each trial
        debug_mode : bool
            If True, shows dragonfly bounding box and logs extra info
        """
        self.user_name = user_name
        self.dragonfly_folder = dragonfly_folder
        self.n_trials = n_trials
        self.image_size = image_size
        self.output_file = output_file

        # Parameter ranges
        self.grain_size_range = grain_size_range
        self.contrast_range = contrast_range
        self.complexity_range = complexity_range
        self.edge_density_options = edge_density_options
        self.dragonfly_scale_range = dragonfly_scale_range
        self.min_border_distance = min_border_distance

        # Display parameters
        self.fixation_duration = fixation_duration
        self.max_trial_time = max_trial_time
        self.show_feedback = show_feedback
        self.debug_mode = debug_mode

        # Load dragonfly images
        self.dragonfly_images = self._load_dragonfly_images()
        if len(self.dragonfly_images) == 0:
            raise ValueError(f"No PNG images found in {dragonfly_folder}")

        print(f"Loaded {len(self.dragonfly_images)} dragonfly images")

        # Trial data
        self.trial_data = []
        self.current_trial = 0
        self.trial_start_time = None
        self.clicked = False

        # Create output file with header
        self._initialize_output_file()

    def _load_dragonfly_images(self):
        """Load all PNG images from dragonfly folder."""
        pattern = os.path.join(self.dragonfly_folder, '*.png')
        image_files = glob.glob(pattern)

        images = {}
        for filepath in image_files:
            filename = os.path.basename(filepath)
            try:
                img = Image.open(filepath).convert('RGBA')
                images[filename] = img
                print(f"  Loaded: {filename} ({img.size[0]}x{img.size[1]})")
            except Exception as e:
                print(f"  Warning: Could not load {filename}: {e}")

        return images

    def _initialize_output_file(self):
        """Create CSV file with header."""
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'user_name',
                'trial_number',
                'timestamp',
                'dragonfly_image',
                'dragonfly_scale',
                'dragonfly_center_x',
                'dragonfly_center_y',
                'dragonfly_width',
                'dragonfly_height',
                'grain_size',
                'contrast',
                'complexity_alpha',
                'edge_density',
                'background_seed',
                'click_x',
                'click_y',
                'click_accuracy_pixels',
                'click_accuracy_normalized',
                'reaction_time_seconds',
                'found',
                'timed_out'
            ])
        print(f"Output file initialized: {self.output_file}")

    def _generate_random_parameters(self):
        """Generate random background parameters for one trial."""
        params = {
            'grain_size': np.random.uniform(*self.grain_size_range),
            'contrast': np.random.uniform(*self.contrast_range),
            'complexity': np.random.uniform(*self.complexity_range),
            'edge_density': np.random.choice(self.edge_density_options),
            'seed': np.random.randint(0, 1000000)
        }
        return params

    def _place_dragonfly(self, background, dragonfly_image, scale_factor):
        """
        Place dragonfly on background at random location.

        Returns
        -------
        composite : ndarray
            Background with dragonfly overlaid
        position : dict
            Contains 'center_x', 'center_y', 'width', 'height', 'bbox'
        """
        # Scale dragonfly
        original_size = dragonfly_image.size
        new_size = (int(original_size[0] * scale_factor),
                    int(original_size[1] * scale_factor))
        dragonfly_scaled = dragonfly_image.resize(new_size, Image.Resampling.LANCZOS)

        # Convert to numpy array
        df_array = np.array(dragonfly_scaled)
        df_rgb = df_array[:, :, :3] / 255.0  # RGB channels
        df_alpha = df_array[:, :, 3] / 255.0  # Alpha channel

        # Choose random position (ensure it fits on screen)
        h, w = df_alpha.shape
        max_x = self.image_size[0] - w - self.min_border_distance
        max_y = self.image_size[1] - h - self.min_border_distance

        x = np.random.randint(self.min_border_distance, max_x)
        y = np.random.randint(self.min_border_distance, max_y)

        # Create composite image
        composite = np.stack([background, background, background], axis=-1)

        # Overlay dragonfly using alpha blending
        for c in range(3):
            composite[y:y + h, x:x + w, c] = (
                    df_rgb[:, :, c] * df_alpha +
                    composite[y:y + h, x:x + w, c] * (1 - df_alpha)
            )

        # Calculate position info
        center_x = x + w // 2
        center_y = y + h // 2

        position = {
            'center_x': center_x,
            'center_y': center_y,
            'width': w,
            'height': h,
            'bbox': (x, y, x + w, y + h)
        }

        return composite, position

    def _calculate_click_accuracy(self, click_x, click_y, position):
        """Calculate how close the click was to dragonfly center."""
        dx = click_x - position['center_x']
        dy = click_y - position['center_y']
        distance = np.sqrt(dx ** 2 + dy ** 2)

        # Normalize by dragonfly size (average of width and height)
        avg_size = (position['width'] + position['height']) / 2
        normalized_distance = distance / avg_size

        return distance, normalized_distance

    def _is_click_on_dragonfly(self, click_x, click_y, position):
        """Check if click is within dragonfly bounding box."""
        x1, y1, x2, y2 = position['bbox']
        return x1 <= click_x <= x2 and y1 <= click_y <= y2

    def _save_trial_data(self, trial_data):
        """Append trial data to CSV file."""
        with open(self.output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                trial_data['user_name'],
                trial_data['trial_number'],
                trial_data['timestamp'],
                trial_data['dragonfly_image'],
                trial_data['dragonfly_scale'],
                trial_data['dragonfly_center_x'],
                trial_data['dragonfly_center_y'],
                trial_data['dragonfly_width'],
                trial_data['dragonfly_height'],
                trial_data['grain_size'],
                trial_data['contrast'],
                trial_data['complexity_alpha'],
                trial_data['edge_density'],
                trial_data['background_seed'],
                trial_data['click_x'],
                trial_data['click_y'],
                trial_data['click_accuracy_pixels'],
                trial_data['click_accuracy_normalized'],
                trial_data['reaction_time_seconds'],
                trial_data['found'],
                trial_data['timed_out']
            ])

    def _onclick(self, event):
        """Handle mouse click events."""
        if event.inaxes and not self.clicked:
            self.clicked = True
            self.click_x = event.xdata
            self.click_y = event.ydata
            self.click_time = time.time()
            plt.close()

    def run_trial(self):
        """Run a single trial."""
        # Generate random parameters
        bg_params = self._generate_random_parameters()
        df_image_name = np.random.choice(list(self.dragonfly_images.keys()))
        df_scale = np.random.uniform(*self.dragonfly_scale_range)

        # Generate background
        background = generate_background(
            width=self.image_size[0],
            height=self.image_size[1],
            grain_size=bg_params['grain_size'],
            contrast=bg_params['contrast'],
            complexity=bg_params['complexity'],
            edge_density=bg_params['edge_density'],
            seed=bg_params['seed']
        )

        # Place dragonfly
        composite, position = self._place_dragonfly(
            background,
            self.dragonfly_images[df_image_name],
            df_scale
        )

        # Show fixation cross
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.text(0.5, 0.5, '+', transform=ax.transAxes,
                fontsize=60, ha='center', va='center', color='red')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.tight_layout()
        plt.draw()
        plt.pause(self.fixation_duration)
        plt.close()

        # Show trial
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(composite)
        ax.axis('off')

        # Debug mode: show bounding box
        if self.debug_mode:
            x1, y1, x2, y2 = position['bbox']
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='red',
                                     facecolor='none', linestyle='--')
            ax.add_patch(rect)
            ax.plot(position['center_x'], position['center_y'],
                    'r+', markersize=20, markeredgewidth=3)

        plt.tight_layout()

        # Setup click handler
        self.clicked = False
        self.click_x = None
        self.click_y = None
        self.trial_start_time = time.time()

        fig.canvas.mpl_connect('button_press_event', self._onclick)

        plt.show(block=True)

        # Calculate results
        if self.clicked:
            reaction_time = self.click_time - self.trial_start_time
            found = self._is_click_on_dragonfly(self.click_x, self.click_y, position)
            click_accuracy, click_accuracy_norm = self._calculate_click_accuracy(
                self.click_x, self.click_y, position
            )
            timed_out = False
        else:
            # Window was closed without clicking
            reaction_time = None
            found = False
            self.click_x = None
            self.click_y = None
            click_accuracy = None
            click_accuracy_norm = None
            timed_out = True

        # Store trial data
        trial_data = {
            'user_name': self.user_name,
            'trial_number': self.current_trial + 1,
            'timestamp': datetime.now().isoformat(),
            'dragonfly_image': df_image_name,
            'dragonfly_scale': df_scale,
            'dragonfly_center_x': position['center_x'],
            'dragonfly_center_y': position['center_y'],
            'dragonfly_width': position['width'],
            'dragonfly_height': position['height'],
            'grain_size': bg_params['grain_size'],
            'contrast': bg_params['contrast'],
            'complexity_alpha': bg_params['complexity'],
            'edge_density': bg_params['edge_density'],
            'background_seed': bg_params['seed'],
            'click_x': self.click_x,
            'click_y': self.click_y,
            'click_accuracy_pixels': click_accuracy,
            'click_accuracy_normalized': click_accuracy_norm,
            'reaction_time_seconds': reaction_time,
            'found': found,
            'timed_out': timed_out
        }

        self._save_trial_data(trial_data)
        self.trial_data.append(trial_data)

        # Show feedback
        if self.show_feedback and not timed_out:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.axis('off')

            if found:
                feedback_text = f"✓ FOUND!\n\nReaction time: {reaction_time:.2f}s"
                color = 'green'
            else:
                feedback_text = f"✗ MISSED\n\nYou clicked outside the dragonfly"
                color = 'red'

            ax.text(0.5, 0.5, feedback_text, transform=ax.transAxes,
                    fontsize=24, ha='center', va='center', color=color,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()
            plt.draw()
            plt.pause(1.0)
            plt.close()

        self.current_trial += 1

        return trial_data

    def run_experiment(self):
        """Run the full experiment."""
        print("\n" + "=" * 60)
        print(f"CAMOUFLAGE DETECTION EXPERIMENT")
        print(f"Participant: {self.user_name}")
        print(f"Trials: {self.n_trials}")
        print("=" * 60)
        print("\nInstructions:")
        print("- You will see a fixation cross (+), then a background")
        print("- Click on the dragonfly as quickly as you can when you find it")
        print("- Try to click accurately on the dragonfly body")
        print("- Press Enter to begin...")
        input()

        # Run trials
        for trial_num in range(self.n_trials):
            print(f"\nTrial {trial_num + 1}/{self.n_trials}")
            trial_data = self.run_trial()

            if trial_data['found']:
                print(f"  ✓ Found in {trial_data['reaction_time_seconds']:.2f}s")
            elif trial_data['timed_out']:
                print(f"  ✗ Trial skipped")
            else:
                print(f"  ✗ Missed (clicked outside dragonfly)")

        # Summary
        completed_trials = [t for t in self.trial_data if not t['timed_out']]
        found_trials = [t for t in completed_trials if t['found']]

        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE!")
        print("=" * 60)
        print(f"Total trials: {len(self.trial_data)}")
        print(f"Completed: {len(completed_trials)}")
        print(f"Found: {len(found_trials)}")
        if len(found_trials) > 0:
            avg_rt = np.mean([t['reaction_time_seconds'] for t in found_trials])
            print(f"Average reaction time: {avg_rt:.2f}s")
            accuracy_rate = len(found_trials) / len(completed_trials) * 100
            print(f"Detection accuracy: {accuracy_rate:.1f}%")
        print(f"\nData saved to: {self.output_file}")
        print("=" * 60)


def main():
    """Run the experiment with example parameters."""
    # Configuration
    USER_NAME = input("Enter participant ID: ").strip()
    DRAGONFLY_FOLDER = input("Enter path to dragonfly images folder: ").strip()
    DRAGONFLY_FOLDER = "D:/camo_dflies"
    # Check if folder exists
    if not os.path.exists(DRAGONFLY_FOLDER):
        print(f"Error: Folder not found: {DRAGONFLY_FOLDER}")
        return

    # Number of trials
    n_trials_input = input("Number of trials (default 50): ").strip()
    N_TRIALS = int(n_trials_input) if n_trials_input else 50

    # Initialize experiment
    experiment = CamouflageExperiment(
        user_name=USER_NAME,
        dragonfly_folder=DRAGONFLY_FOLDER,
        n_trials=N_TRIALS,
        image_size=(800, 800),
        output_file=f'camouflage_data_{USER_NAME}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',

        # Background parameter ranges
        #grain_size_range=(10, 60),  # Fine to coarse texture
        #contrast_range=(0.2, 0.8),  # Low to high contrast
        #complexity_range=(0.5, 1.5),  # Natural variation
        grain_size_range=(10, 60),  # Fine to coarse texture
        contrast_range=(0.2, 0.8),  # Low to high contrast
        complexity_range=(0.5, 1.5),  # Natural variation
        edge_density_options=['auto', 'low', 'medium', 'high'],

        # Dragonfly parameters
        dragonfly_scale_range=(0.5, 0.5),  # Size variation
        min_border_distance=100,  # Keep away from edges

        # Display parameters
        fixation_duration=0.5,  # 500ms fixation
        max_trial_time=30.0,  # 30s timeout
        show_feedback=True,  # Show feedback after each trial
        debug_mode=False  # Set to True to see bounding boxes
    )

    # Run experiment
    experiment.run_experiment()

if __name__ == '__main__':
    main()