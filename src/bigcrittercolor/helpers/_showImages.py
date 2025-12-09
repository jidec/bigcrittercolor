import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random


def _showImages(show, images, titles=None, maintitle=None, list_cmaps=None, grid=False,
                num_cols=3, figsize=(10, 10), title_fontsize='auto', sample_n=None,
                save_folder=None, title_wrap=True, title_max_chars=30):
    '''
    Shows a grid of images with automatically adjusted titles to prevent overlap.

    Parameters:
    ----------
    images: list
        List of the images to be displayed.
    titles: list or None
        Optional list of titles to be shown for each image.
    list_cmaps: list or None
        Optional list of cmap values for each image.
    grid: boolean
        If True, show a grid over each image
    num_cols: int
        Number of columns to show.
    figsize: tuple of width, height
        Value to be passed to pyplot.figure()
    title_fontsize: int or 'auto'
        If 'auto', automatically calculate based on subplot size.
        Otherwise, value to be passed to set_title().
    sample_n: int or None
        If specified, randomly sample n images to display
    save_folder: str or None
        Folder path to save the figure
    title_wrap: boolean
        If True, wrap long titles to multiple lines
    title_max_chars: int
        Maximum characters per line when wrapping titles
    '''

    if show:
        # Convert to lists if needed
        if not isinstance(images, list):
            images = [images]
        if titles is not None and not isinstance(titles, list):
            titles = [titles]

        list_images = images
        list_titles = titles

        # Sample n images if specified
        if sample_n is not None and sample_n < len(list_images):
            indices = random.sample(range(len(list_images)), sample_n)
            list_images = [list_images[i] for i in indices]
            if list_titles is not None:
                list_titles = [list_titles[i] for i in indices]

        # Convert images to RGB
        for index, img in enumerate(list_images):
            if len(img.shape) == 4:
                img = img[:, :, :3]
                img = img.astype(np.uint8)
            list_images[index] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Assertions
        assert isinstance(list_images, list)
        assert len(list_images) > 0
        assert isinstance(list_images[0], np.ndarray)

        if list_titles is not None:
            assert isinstance(list_titles, list)
            assert len(list_images) == len(list_titles), '%d imgs != %d titles' % (len(list_images), len(list_titles))

        if list_cmaps is not None:
            assert isinstance(list_cmaps, list)
            assert len(list_images) == len(list_cmaps), '%d imgs != %d cmaps' % (len(list_images), len(list_cmaps))

        num_images = len(list_images)
        num_cols = min(num_images, num_cols)
        num_rows = int(num_images / num_cols) + (1 if num_images % num_cols != 0 else 0)

        # Create figure and subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

        # Create list of axes
        if isinstance(axes, np.ndarray):
            list_axes = list(axes.flat)
        else:
            list_axes = [axes]

        # Calculate automatic font size if needed
        if title_fontsize == 'auto':
            # Base font size on subplot dimensions
            subplot_width = figsize[0] / num_cols
            subplot_height = figsize[1] / num_rows

            # Use smaller dimension for font size calculation
            subplot_size = min(subplot_width, subplot_height)

            # Calculate font size (adjust multiplier as needed)
            # Smaller subplots get smaller fonts
            calculated_fontsize = max(8, min(16, int(subplot_size * 2.5)))
        else:
            calculated_fontsize = title_fontsize

        # Display images
        for i in range(num_images):
            img = list_images[i]
            title = list_titles[i] if list_titles is not None else None
            cmap = list_cmaps[i] if list_cmaps is not None else (None if img_is_color(img) else 'gray')

            list_axes[i].imshow(img, cmap=cmap)

            if title is not None:
                # Wrap long titles if enabled
                if title_wrap and len(title) > title_max_chars:
                    wrapped_title = wrap_title(title, title_max_chars)
                else:
                    wrapped_title = title

                # Set title with calculated font size
                list_axes[i].set_title(wrapped_title, fontsize=calculated_fontsize,
                                       pad=calculated_fontsize * 0.5)  # Add padding based on font size

            list_axes[i].grid(grid)

        # Hide unused subplots
        for i in range(num_images, len(list_axes)):
            list_axes[i].set_visible(False)

        # Remove ticks
        for i in range(len(list_axes)):
            list_axes[i].set_xticks([])
            list_axes[i].set_yticks([])

        # Adjust layout with better spacing
        if maintitle is not None:
            # Make room for main title
            fig.suptitle(maintitle, fontsize=calculated_fontsize * 1.5 if title_fontsize == 'auto' else 30)
            # Adjust spacing to prevent overlap
            plt.subplots_adjust(top=0.92, bottom=0.02, left=0.02, right=0.98,
                                hspace=0.15, wspace=0.05)
        else:
            # No main title, can use more space
            plt.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98,
                                hspace=0.10, wspace=0.05)

        # Save if specified
        if save_folder is not None:
            filename = generate_unique_filename(save_folder, "plot", ".jpg")
            plt.savefig(os.path.join(save_folder, filename), bbox_inches='tight', dpi=100)

        plt.show()


def wrap_title(title, max_chars):
    """Wrap title text to multiple lines."""
    words = title.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        if current_length + len(word) + len(current_line) > max_chars:
            if current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                # Single word longer than max_chars
                lines.append(word)
                current_line = []
                current_length = 0
        else:
            current_line.append(word)
            current_length += len(word)

    if current_line:
        lines.append(' '.join(current_line))

    return '\n'.join(lines)


def img_is_color(img):
    """Check if an image is color or grayscale."""
    if len(img.shape) == 3:
        # Check the color channels to see if they're all the same.
        c1, c2, c3 = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        if (c1 == c2).all() and (c2 == c3).all():
            return True
    return False


def generate_unique_filename(directory, base_filename, extension):
    """Generate a unique filename in the specified directory."""
    i = 1
    unique_filename = f"{base_filename}{extension}"
    while os.path.exists(os.path.join(directory, unique_filename)):
        unique_filename = f"{base_filename}_{i}{extension}"
        i += 1
    return unique_filename


# Alternative: Use constrained_layout for automatic spacing (matplotlib >= 3.0)
def _showImages_constrained(show, images, titles=None, maintitle=None, list_cmaps=None,
                            grid=False, num_cols=3, figsize=(10, 10), sample_n=None,
                            save_folder=None):
    '''
    Alternative version using constrained_layout for better automatic spacing.
    Requires matplotlib >= 3.0
    '''

    if show:
        # [Same preprocessing code as above until fig creation]
        if not isinstance(images, list):
            images = [images]
        if titles is not None and not isinstance(titles, list):
            titles = [titles]

        list_images = images
        list_titles = titles

        if sample_n is not None and sample_n < len(list_images):
            indices = random.sample(range(len(list_images)), sample_n)
            list_images = [list_images[i] for i in indices]
            if list_titles is not None:
                list_titles = [list_titles[i] for i in indices]

        for index, img in enumerate(list_images):
            if len(img.shape) == 4:
                img = img[:, :, :3]
                img = img.astype(np.uint8)
            list_images[index] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        num_images = len(list_images)
        num_cols = min(num_images, num_cols)
        num_rows = int(num_images / num_cols) + (1 if num_images % num_cols != 0 else 0)

        # Use constrained_layout for automatic spacing
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize,
                                 constrained_layout=True)

        if isinstance(axes, np.ndarray):
            list_axes = list(axes.flat)
        else:
            list_axes = [axes]

        # Auto-calculate font size
        subplot_size = min(figsize[0] / num_cols, figsize[1] / num_rows)
        fontsize = max(8, min(16, int(subplot_size * 2.5)))

        for i in range(num_images):
            img = list_images[i]
            title = list_titles[i] if list_titles is not None else None
            cmap = list_cmaps[i] if list_cmaps is not None else (None if img_is_color(img) else 'gray')

            list_axes[i].imshow(img, cmap=cmap)
            if title is not None:
                list_axes[i].set_title(title, fontsize=fontsize)
            list_axes[i].grid(grid)
            list_axes[i].set_xticks([])
            list_axes[i].set_yticks([])

        for i in range(num_images, len(list_axes)):
            list_axes[i].set_visible(False)

        if maintitle is not None:
            fig.suptitle(maintitle, fontsize=fontsize * 1.5)

        if save_folder is not None:
            filename = generate_unique_filename(save_folder, "plot", ".jpg")
            plt.savefig(os.path.join(save_folder, filename), bbox_inches='tight')

        plt.show()