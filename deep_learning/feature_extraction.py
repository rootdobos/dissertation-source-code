import sys
from .loaders.image_batch_loader import load_batch_from_dir
from .models.feature_extractors.resnet import Resnet50, Resnext50
from .models.feature_extractors.efficientnet import EfficientNetB1, EfficientNetB3, EfficientNetB4, EfficientNetV2
from .models.transforms.transforms import transform_resnet, transform_efficientNetB3, transform_efficientNet, transform_resnext, transform_efficientNetB4, transform_efficientNetV2
import torch
from tqdm import tqdm

from torchvision import datasets
from torch.utils.data import DataLoader

import pandas as pd
import os


device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


def extract_features(model, transform, slide_tiles_path):
    batch = load_batch_from_dir(slide_tiles_path, transform)
    batch = batch.to(device)

    with torch.no_grad():
        features = model(batch)
    return features


if __name__ == '__main__':
    args = sys.argv

    data_dir = args[1]
    csv_path = args[2]
    output_dir = args[3]
    extractor = args[4]

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    if extractor == "resnet":
        model = Resnet50()
        transform = transform_resnet
    elif extractor == "resnext":
        model = Resnext50()
        transform = transform_resnext
    elif extractor == "effnetv2":
        model = EfficientNetV2()
        transform = transform_efficientNetV2
    elif extractor == "effnetb4":
        model = EfficientNetB4()
        transform = transform_efficientNetB4
    elif extractor == "effnetb3":
        model = EfficientNetB3()
        transform = transform_efficientNetB3
    else:
        model = EfficientNetB1()
        transform = transform_efficientNet

    model.eval()
    model.to(device)
    with torch.inference_mode():
        for slide in tqdm(list(df['image_id'])):
            output_file_name = os.path.join(
                output_dir, "{}.pt".format(slide))
            if os.path.exists(output_file_name):
                continue
            features = extract_features(
                model, transform, os.path.join(data_dir, slide, "tiles"))
            torch.save(features, output_file_name)
