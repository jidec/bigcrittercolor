from PIL import Image
import os

def convertTIFs(source_folder, destination_folder, downscale_factor):
    """
    Converts TIF images in the source folder to downscaled PNG images in the destination folder.

    Parameters:
    - source_folder: Path to the folder containing the TIF images.
    - destination_folder: Path where the downscaled PNG images will be saved.
    - downscale_factor: Factor by which the images will be downscaled.
    """

    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    # Loop through all TIFF files in the source folder
    for file_name in os.listdir(source_folder):
        if file_name.lower().endswith('.tif') or file_name.lower().endswith('.tiff'):
            # Construct full file path
            file_path = os.path.join(source_folder, file_name)

            # Open the image
            with Image.open(file_path) as img:
                # Compute the new size
                new_size = (img.width // downscale_factor, img.height // downscale_factor)

                # Resize the image
                img_resized = img.resize(new_size, Image.ANTIALIAS)

                # Convert the file name to PNG
                png_file_name = os.path.splitext(file_name)[0] + '.png'

                # Construct the full path for the destination file
                png_file_path = os.path.join(destination_folder, png_file_name)

                # Save the resized image as PNG
                img_resized.save(png_file_path, 'PNG')

    print("Conversion completed.")

# Example usage:
convertTIFs('D:/GitProjects/new_wings_project/tifs', 'D:/GitProjects/new_wings_project/pngs', 5)