import sys
from loaders.image_batch_loader import load_batch_from_dir
from models.feature_extractors.resnet import Resnet50
from models.feature_extractors.efficientnet import EfficientNetB1
from models.transforms.transforms import transform_resnet, transform_efficientNet
import torch
from tqdm import tqdm

from torchvision import datasets
from torch.utils.data import DataLoader

import pandas as pd
import os



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def extract_features(model,transform, slide_tiles_path):
    batch = load_batch_from_dir(slide_tiles_path,transform)
    batch=batch.to(device)

    with torch.no_grad():
        features=model(batch)
    return features

if __name__ == '__main__':
    args=sys.argv

    data_dir=args[1]
    csv_path=args[2]
    output_dir=args[3]
    extractor= args[4]
    
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    if extractor=="resnet":
        model=Resnet50()
        transform=transform_resnet
    else:
        model=EfficientNetB1()
        transform=transform_efficientNet
    
    model.eval()
    model.to(device)
    for slide in  tqdm(list(df['image_id'])):
        features=extract_features(model,transform,os.path.join(data_dir,slide,"tiles"))
        torch.save(features,os.path.join(output_dir,"{}.pt".format(slide)))
