import torch
from torch import nn

from torchvision.models.efficientnet import (efficientnet_b1, efficientnet_b4, efficientnet_v2_s, EfficientNet_B4_Weights, EfficientNet_V2_S_Weights,
                                             efficientnet_b3, EfficientNet_B3_Weights)


class EfficientNetB1(nn.Module):
    def __init__(self):
        super().__init__()
        model = efficientnet_b1(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        return self.model(x)


class EfficientNetB4(nn.Module):
    def __init__(self):
        super().__init__()
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1
        model = efficientnet_b4(weights=weights)
        self.model = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        return self.model(x)


class EfficientNetB3(nn.Module):
    def __init__(self):
        super().__init__()
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1
        model = efficientnet_b3(weights=weights)
        self.model = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        return self.model(x)


class EfficientNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
        model = efficientnet_v2_s(weights=weights)
        self.model = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        return self.model(x)
