import torch
from torchvision import transforms
from PIL import Image
import os


def load_batch_from_dir(image_dir, transform):
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]


    image_tensors = []

    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        
        # Open and preprocess the image
        image = Image.open(img_path).convert("RGB")
        image = transform(image)  # Shape: (3, 224, 224)
        
        # Append to list
        image_tensors.append(image)

    image_batch = torch.stack(image_tensors)
    return image_batch
