import os
import shutil

def clearFolder(folder_path):
    """
    Deletes all files and folders within the specified folder,
    leaving the folder itself intact. Asks for user confirmation
    before proceeding.

    :param folder_path: Path to the folder to clear.
    """
    # Extract the folder name for display in the confirmation message
    folder_name = os.path.basename(folder_path)

    # User confirmation
    response = input(f'Are you sure you want to clear {folder_name}? (yes/no): ')
    if response.lower() != 'yes':
        print("Operation cancelled.")
        return

    for item_name in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item_name)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)  # Removes files and links
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Removes directories and their contents
        except Exception as e:
            print(f'Failed to delete {item_path}. Reason: {e}')