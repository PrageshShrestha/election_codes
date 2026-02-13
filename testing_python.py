from pathlib import Path
from PIL import Image
import time
folder = Path("voter_info")
import cv2
import os
import subprocess
import csv



def get_text(image):
    main_model = "/home/pragesh-shrestha/Desktop/nishant_sir/allenai_olmOCR-2-7B-1025-Q4_K_M.gguf"
    mmproj_file = "/home/pragesh-shrestha/Desktop/nishant_sir/mmproj-allenai_olmOCR-2-7B-1025-f16.gguf"
    test_image = image
    prompt_file = """
            You are an OCR transcription engine.

    Your task is strict character transcription.

    Rules you MUST follow:
    - Output only the characters visible in the image.
    - Preserve exact order.
    - Preserve spaces and line breaks.
    - Do NOT translate.
    - Do NOT explain.
    - Do NOT identify language.
    - Do NOT add quotes.
    - Do NOT add punctuation that is not visible.
    - Do NOT add headings.
    - Do NOT say "The text is".
    - Do NOT describe the image.
    - If unsure about a character, copy it as seen.

    Your response must contain ONLY the raw text.

    <END_OF_INSTRUCTIONS>
                                                                        
            """


    # Build CLI command
    cmd = [
        "./llama.cpp/build/bin/llama-mtmd-cli",
        "-m", main_model,
        "--mmproj", mmproj_file,
        "--image", test_image,
        "-p", f"@{prompt_file}",
        "--temp", "0.1",
        "-n", "512",
        "-c", "4096"
    ]



    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode == 0:
        
        return result.stdout

    else:
        return 0





def parse_actual(image_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image.")
        return

    # Create the output directory if it doesn't exist
    output_folder = "temp_ocr_img"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Dictionary of regions [x1, y1, x2, y2]
    regions = {
        "voter_id": [0, 0, 743, 141],
        "name": [0, 141, 743, 275],
        "age_gender": [0, 275, 743, 416],
        "parent_name": [0, 416, 947, 518],
        "spouse": [746, 300, 1541, 390],
        "picture": [1584, 8, 2019, 549]
    }
    voter_id = 0
    listed_output = []
    for label, coords in regions.items():
        x1, y1, x2, y2 = coords
        
        # Crop using slicing: image[y1:y2, x1:x2]
        crop = img[y1:y2, x1:x2]
        
        # Save the file
        if label == "picture":
            save_path = os.path.join("voter_images/", f"{voter_id}.jpg")
            cv2.imwrite(save_path, crop)
            voter_id = 0
            listed_output.append(save_path)
            return listed_output
        save_path = os.path.join(output_folder, f"{label}.jpg")
        cv2.imwrite(save_path, crop)
        text = get_text(save_path)
        listed_output.append(text)
        if label == "voter_id":
            listed_output = []
            voter_id = text









def parse_extra(image_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image.")
        return

    # Create the output directory if it doesn't exist
    output_folder = "temp_ocr_img"
    

    

        # district : [0,0,847,187]
        # legislative_area:[2167,11,2305,165]
        # provincial_area:[2926,33,3080,154]
        # municipality:[3388,17,4780,154]
        # ward:[5066,17,5269,157]
        # booth:[5742,11,7282,187]




    regions = {
    "district": [0, 0, 847, 187],
    "legislative_area": [2167, 11, 2305, 165],
    "provincial_area": [2926, 33, 3080, 154],
    "municipality": [3388, 17, 4780, 154],
    "ward": [5066, 17, 5269, 157],
    "booth": [5742, 11, 7282, 187]
        }
    
    listed_output = []
    for label, coords in regions.items():
        x1, y1, x2, y2 = coords
        
        # Crop using slicing: image[y1:y2, x1:x2]
        crop = img[y1:y2, x1:x2]
        
        
        save_path = os.path.join(output_folder, f"{label}.jpg")
        cv2.imwrite(save_path, crop)
        text = get_text(save_path)
        listed_output.append(text)
    return listed_output
# Run the function
# parse_actual("your_voter_card.jpg")


def clear_temp_folder(folder_path):
    path = Path(folder_path)
    if path.exists():
        for file in path.glob("*"):
            if file.is_file():
                file.unlink()
with open("output.csv", "w") as csvfile:
    fieldnames = ["voter_id","name","age_gender","parent_name","spouse","picture","district","legislative_area","provincial_area","municipality","ward","booth"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for image_path in folder.glob("*.*"):
        start = time.time_ns()
        
        actual_infos = []
        name = image_path.name
        name_only = Path(name).stem
        img1 = image_path 
        img2_name = f"{name_only}_extra.jpg"
        img2 = f"voter_extra/{img2_name}"
        
        var1 = parse_actual(img1)
        var2 = parse_extra(img2)
        combined_list = var1 + var2 
        
        # Map the combined list to the fieldnames dictionary
        row_dict = dict(zip(fieldnames, combined_list))
        
        # Use writerow (singular) to save one voter at a time
        writer.writerow(row_dict)
        clear_temp_folder("temp_ocr_img")
        end = time.time_ns()

        # Calculate duration
        duration_ns = end - start
       
        duration_sec = duration_ns / 1_000_000_000  # Convert to seconds

        print(f"Processed {image_path.name} with time of {duration_sec} seconds")
