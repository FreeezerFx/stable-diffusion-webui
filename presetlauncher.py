import requests
import base64
import os
from datetime import datetime
import time
import json
import torch
import numpy as np

def timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")


# Define API URL
API_URL = "http://127.0.0.1:7861/sdapi/v1/txt2img"
# API document http://127.0.0.1:7861/docs#/

# Create output directory if it doesn't exist
output_dir = "generated_images"
latent_dir = "latent_vectors"
intermediate_dir = "intermediate_steps"
intermediate_steps = [5,15,25]
os.makedirs(output_dir, exist_ok=True)
os.makedirs(latent_dir, exist_ok=True)
os.makedirs(intermediate_dir, exist_ok=True)


# Define payload
payload = {
    "prompt": "One perfectly centered white square on one solid black background, minimalistic, high contrast, no noise, sharp edges, simple composition, no shading, no texture, pure black and white",
    "seed": 4230199263,
    "steps": 30,
    "width": 512,
    "height": 512,
    "cfg_scale": 10,
    "sampler_name": "DPM++ 2M Karras",
    "return_latent": True,
    "save_images": True,
    "save_steps": intermediate_steps
}

# Send request
response = requests.post(API_URL, json=payload)

if response.status_code == 200:
    result = response.json()
    
    print(result.keys())

    # Loop through each generated image in the batch
    for idx, image_data in enumerate(result['images']):
        # Decode Base64 image data
        image_bytes = base64.b64decode(image_data)
        
        # Generate filename
        filename = os.path.join(output_dir, f"output-{timestamp()}-{idx+1}.png")
        
        # Save the image
        with open(filename, "wb") as img_file:
            img_file.write(image_bytes)
        
        print(f"Image saved as {filename}")
else:
    print("Error:", response.text)
