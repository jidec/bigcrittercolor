import os
from PIL import Image
from PIL.ExifTags import TAGS

# some is present but mega rare

def get_camera_model(exif_data):
    """Extract camera model from EXIF data."""
    if exif_data is None:
        return None
    for tag, value in exif_data.items():
        tag_name = TAGS.get(tag, tag)
        if tag_name == 'Model':  # Check for the 'Model' tag
            return value
    return None


def check_images_for_camera_model(folder_path):
    """Check all images in the folder for camera model metadata."""
    images_with_camera_model = {}

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('png', 'jpg', 'jpeg', 'tiff', 'bmp')):
            file_path = os.path.join(folder_path, filename)
            try:
                with Image.open(file_path) as img:
                    exif_data = img._getexif()  # Get EXIF metadata
                    camera_model = get_camera_model(exif_data)
                    if camera_model:
                        images_with_camera_model[filename] = camera_model
            except Exception as e:
                images_with_camera_model[filename] = f"Error: {e}"

    return images_with_camera_model


# Usage
folder_path = "D:/bcc/ringtails/all_images"
results = check_images_for_camera_model(folder_path)

# Print results
for image, camera_model in results.items():
    print(f"{image}: {camera_model}")