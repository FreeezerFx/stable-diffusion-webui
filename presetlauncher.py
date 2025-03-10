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

# Create output directory if it doesn't exist
output_dir = "generated_images"
latent_dir = "latent_vectors"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(latent_dir, exist_ok=True)

# Define payload
payload = {
    "prompt": "One perfectly centered white square on one solid black background, minimalistic, high contrast, no noise, sharp edges, simple composition, no shading, no texture, pure black and white",
    "seed": 4230199263,
    "steps": 30,
    "width": 512,
    "height": 512,
    "cfg_scale": 10,
    "sampler_name": "DPM++ 2M Karras",
    "return_latent": True
}

# Send request
response = requests.post(API_URL, json=payload)

if response.status_code == 200:
    result = response.json()
    
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
    if 'latents' in result:
        latent_vectors = result['latents']  # Assuming it's a list of latent tensors
        latent_tensor = torch.tensor(latent_vectors)

        latent_filename = os.path.join(latent_dir, f"latent-{timestamp()}.pt")
        torch.save(latent_tensor, latent_filename)

        print(f"Latent vector saved as {latent_filename}")
else:
    print("Error:", response.text)
