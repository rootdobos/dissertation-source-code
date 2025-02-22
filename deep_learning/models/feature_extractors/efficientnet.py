import torch
from torch import nn

from torchvision.models.efficientnet import efficientnet_b1


class EfficientNetB1(nn.Module):
    def __init__(self):
        super().__init__()
        model=efficientnet_b1(pretrained=True)
        self.model=nn.Sequential(*list(model.children())[:-1])

    def forward(self,x):
        return self.model(x)

