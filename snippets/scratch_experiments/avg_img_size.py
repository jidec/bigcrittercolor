import os


def calculate_avg_image_size(folder_path):
    # List to store file sizes
    image_sizes = []

    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Check if it's a file and ends with a common image extension
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
            image_sizes.append(os.path.getsize(file_path))

    # Calculate the average size if there are images
    if image_sizes:
        avg_size = sum(image_sizes) / len(image_sizes)
        return avg_size
    else:
        return None


# Example usage
folder_path = 'D:/bcc/ringtails/all_images'
avg_filesize = calculate_avg_image_size(folder_path)
if avg_filesize:
    print(f"Average image file size: {avg_filesize / 1024:.2f} KB")
else:
    print("No images found in the folder.")

import os

def print_directory_tree(startpath, indent_level=0):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f'{indent}{os.path.basename(root)}/')
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            print(f'{sub_indent}{f}')

# Call the function with the folder you want to visualize
print_directory_tree("D:/bcc/new_random_beetles")