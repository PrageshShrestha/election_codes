import os
import pandas as pd

# 1. Setup paths
image_folder = 'voter_images/'  # The path to your folder containing the .jpg files
en_to_dev_map = str.maketrans('0123456789', '०१२३४५६७८९')

def get_cleaned_name(filename):
    """Returns the new filename without spaces and with Devanagari digits."""
    # Remove spaces
    new_name = filename.replace(" ", "")
    # Translate digits
    new_name = new_name.translate(en_to_dev_map)
    return new_name

# 2. Process the files
if not os.path.exists(image_folder):
    print(f"Error: The folder '{image_folder}' was not found.")
else:
    count = 0
    # List all files in the folder
    for filename in os.listdir(image_folder):
        # We only want to rename the files that actually need changing
        new_filename = get_cleaned_name(filename)
        
        if new_filename != filename:
            old_path = os.path.join(image_folder, filename)
            new_path = os.path.join(image_folder, new_filename)
            
            # Perform the actual rename
            try:
                os.rename(old_path, new_path)
                count += 1
            except FileExistsError:
                print(f"Skip: '{new_filename}' already exists.")
            except Exception as e:
                print(f"Error renaming '{filename}': {e}")

    print(f"--- Finished! Total files renamed: {count} ---")