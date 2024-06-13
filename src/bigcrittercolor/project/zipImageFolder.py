import os
import zipfile

def zipImageFolder(image_folder, output_zip_path):
    # Create a zip file for writing with compression
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        # Walk through the folder
        for root, _, files in os.walk(image_folder):
            for file in files:
                # Construct the full file path
                file_path = os.path.join(root, file)
                # Add file to zip archive
                # The arcname argument avoids storing the full path in the zip file
                z.write(file_path, arcname=os.path.relpath(file_path, start=image_folder))
                print(f"Added {file_path} to the zip archive")  # Optional, for progress tracking

# Example usage
#image_folder = 'D:/bcc/butterflies/all_images'
#output_zip_path = 'D:/bcc/butterflies/all_images.zip'

#zipImageFolder(image_folder,output_zip_path)