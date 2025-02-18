import imageio
import os
from PIL import Image
import numpy as np

# Folder containing PNG images
image_folder = './gifpngs'

# Get list of image files
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])

# List to store images
images = []

# Read images and convert to numpy arrays
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    img = Image.open(image_path)
    
    # Ensure the image is in RGB mode (or convert to 'RGBA' if needed)
    img = img.convert('RGBA')
    
    # Convert image to numpy array and add it to the images list
    images.append(np.array(img))

# Create the MP4 video
output_video = 'output_video.mp4'
fps = 60  # Set frames per second (adjust as needed)
duration_per_frame = 3  # Duration in seconds for each frame (each image)

with imageio.get_writer(output_video, mode='I', fps=fps, codec='libx264') as writer:
    for img in images:
        writer.append_data(img)

print(f"MP4 video created: {output_video}")
