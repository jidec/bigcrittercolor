import csv
import os
from PIL import Image
import matplotlib.pyplot as plt
from bigcrittercolor.helpers import _readBCCImgs, _getBCCIDs


def manualCodeSegments(img_ids=None, type="segment", codings=None, data_folder=''):
    """
    Presents images for manual coding and updates results in a CSV.

    Args:
    - img_ids (list): A list of unique image IDs.
    - type (str): The type of image (e.g., "segment").
    - codings (list): A list of dictionaries specifying coding categories and options.
      Example: [{"name": "lighting", "options": ["overexposed", "underexposed", "normal"]}]
    - data_folder (str): The folder containing the images and where to save the CSV.

    This function checks existing codings in a CSV for each image ID,
    updates missing codings, and rewrites the CSV file after every change.
    """

    if img_ids is None:
        img_ids = _getBCCIDs(type=type, data_folder=data_folder)

    # Load images
    imgs = _readBCCImgs(img_ids, type=type, data_folder=data_folder)

    # Load existing codings into a dictionary
    existing_codings = {}
    csv_location = os.path.join(data_folder, "manual_codings.csv")
    if os.path.exists(csv_location):
        with open(csv_location, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_id = row["id"]
                existing_codings[img_id] = {key: row[key] for key in row.keys()}

    # Create or update codings
    all_coding_names = [coding["name"] for coding in codings]
    for img, img_id in zip(imgs, img_ids):
        # Use existing row or create a new one
        row_data = existing_codings.get(img_id, {"id": img_id})

        # Check for missing codings
        missing_codings = [coding for coding in codings if coding["name"] not in row_data or not row_data[coding["name"]]]

        if not missing_codings:
            continue  # Skip if all codings are already present

        # Display the image
        if isinstance(img, str):
            img = Image.open(img)
        elif hasattr(img, 'shape'):  # If it's a NumPy array, check channels
            if img.shape[-1] == 3:  # Convert BGR to RGB if necessary
                img = img[..., ::-1]  # Reverse the channel order

        plt.imshow(img)
        plt.axis('off')
        plt.show()

        # Iterate through missing codings
        for coding in missing_codings:
            coding_name = coding["name"]
            options = coding["options"]

            # Print coding options
            print(f"Coding category: {coding_name}")
            print("Options:")
            for i, option in enumerate(options):
                print(f"{i}: {option}")

            # Get user input
            while True:
                user_input = input(f"Enter code number for {coding_name} (0-{len(options) - 1}) for image {img_id}: ")
                if user_input.isdigit() and int(user_input) in range(len(options)):
                    selected_code = options[int(user_input)]
                    row_data[coding_name] = selected_code
                    print(f"Coded image {img_id} - {coding_name}: '{selected_code}'.")
                    break
                else:
                    print(f"Invalid input. Please enter a number between 0 and {len(options) - 1}.")

        # Update the dictionary
        existing_codings[img_id] = row_data

        # Rewrite the entire CSV file after every update
        with open(csv_location, mode='w', newline='') as f:
            fieldnames = ["id"] + all_coding_names
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_codings.values())

    print("All images processed and codings updated.")

#codings = [
#    {"name": "lighting", "options": ["overexposed", "underexposed", "normal"]},
#    {"name": "background", "options": ["natural", "artificial"]}
#    ,{"name": "focus", "options": ["sharp", "blurry"]},
#]

#manualCodeSegments(type="segment", codings=codings, data_folder="D:/bcc/ringtails")