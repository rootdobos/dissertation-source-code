import os
import torch

from ..models.feature_extractors.resnet import Resnet50, Resnext50
from ..models.feature_extractors.efficientnet import EfficientNetB1, EfficientNetV2, EfficientNetB3, EfficientNetB4
from ..models.transforms.transforms import transform_resnet, transform_efficientNet, transform_resnext, transform_efficientNetV2, transform_efficientNetB3, transform_efficientNetB4
from ..feature_extraction import extract_features
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


class FeatureExtractionService():

    @staticmethod
    def extract_features(extractor, slide_id, tile_directory, feature_directory):
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

        input_dir = os.path.join(tile_directory, slide_id)
        features = extract_features(model, transform, input_dir)
        torch.save(features, os.path.join(
            feature_directory, "{}.pt".format(slide_id)))
