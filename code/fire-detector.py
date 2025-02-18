from picamera2 import Picamera2
import cv2
import numpy as np
import time

import os
import git
import requests
from requests.auth import HTTPBasicAuth
import base64

def detect_fire(image):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Updated fire color range to include more red hues
    lower_bound = np.array([165, 20, 20])   # Lower bound for red
    upper_bound = np.array([200, 255, 255])  # Upper bound for red

    # Create mask to detect fire-like pixels
    fire_mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Apply mask to original image
    fire_detected = cv2.bitwise_and(image, image, mask=fire_mask)

    return fire_detected, fire_mask

def calculate_red_percentage(mask):
    total_pixels = mask.size
    red_pixels = cv2.countNonZero(mask)
    red_percentage = (red_pixels / total_pixels) * 100
    return red_percentage

def analyze_fire_spread(mask):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return "No fire detected"

    # Calculate the total area of fire
    fire_area = sum(cv2.contourArea(cnt) for cnt in contours)

    # Get the bounding box for the largest fire region
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    bounding_box_area = w * h

    # Determine spread vs. concentration
    spread_ratio = fire_area / bounding_box_area

    if spread_ratio < 0.5:
        return "Fire is spread out"
    else:
        return "Fire is concentrated"

def get_file_sha(repo_owner, repo_name, file_path, github_token):
    # GitHub API endpoint to get the file content metadata
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}"

    # Send a GET request to retrieve file metadata
    response = requests.get(url, headers={"Authorization": f"Bearer {github_token}"})

    if response.status_code == 200:
        # Extract the SHA from the response JSON
        file_data = response.json()
        return file_data['sha']
    else:
        print(f"Failed to get file SHA: {response.status_code}, {response.text}")
        return None
        
def upload_to_github(image_path, repo_owner, repo_name, file_path, github_token):
    sha = get_file_sha(repo_owner, repo_name, file_path, github_token)
    if not sha:
        print(f"File SHA is empty")
        return
        
    # Read image as binary
    with open(image_path, 'rb') as f:
        image_data = f.read()

    # Encode the image in base64
    encoded_image = base64.b64encode(image_data).decode()

    # GitHub API endpoint for creating a file
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}"

    # Create payload for the request
    payload = {
        "message": "Upload captured fire image",
        "content": encoded_image,
        "sha": sha,  # Provide the existing file SHA to update it
        "branch": "main"  # Or your desired branch
    }

    # Send a PUT request to GitHub API to upload the file
    response = requests.put(
        url,
        headers={"Authorization": f"Bearer {github_token}"},
        json=payload
    )

    if response.status_code == 201 or response.status_code == 200:
        print(f"Image uploaded successfully to GitHub at {file_path}")
    else:
        print(f"Failed to upload image: {response.status_code}, {response.text}")

# Add your GitHub credentials and repository details
repo_owner = "ChinmayKamjith1"  # GitHub username
repo_name = "CubeSat-Repo"  # Repository name
file_path = "captured_images1/image.jpg"  # Path where image will be saved in the repo
github_token = "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"  # Personal access token for authentication

# Initialize the camera
picam = Picamera2()

# Configure and start the camera
picam.configure(picam.create_preview_configuration())
picam.start()

try:
    while True:
        print("Running the process...")
        time.sleep(2)  # Allow time for auto-exposure adjustment

        # Capture an image
        image_path = "image.jpg"
        picam.capture_file(image_path)
        print("Image captured!")

        # Load captured image
        image = cv2.imread(image_path)

        if image is None:
            print("Error: Image not found.")
            continue  # Skip to next iteration

        # Detect fire
        fire_detected, mask = detect_fire(image)

        # Calculate and print red percentage
        red_percentage = calculate_red_percentage(mask)
        print(f"Red percentage: {red_percentage:.2f}%")

        # Analyze and print fire spread
        fire_spread_status = analyze_fire_spread(mask)
        print(fire_spread_status)

        # Upload image to GitHub if conditions are met
        if red_percentage > 10 and fire_spread_status == "Fire is concentrated":
            upload_to_github(image_path, repo_owner, repo_name, file_path, github_token)
        else:
            print(f"Image didnt pass the fire detect criteria")

        # Wait for 30 seconds before capturing the next image
        time.sleep(30)
except KeyboardInterrupt:
    print("Process interrupted. Exiting the loop.")
    
picam.stop()
# Display results
# cv2.imshow("Original Image", image)
# cv2.imshow("Fire Mask", mask)
# cv2.imshow("Detected Fire", fire_detected)

cv2.waitKey(0)
cv2.destroyAllWindows()

