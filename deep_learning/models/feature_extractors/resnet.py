import torch
from torch import nn
from torchvision.models.resnet import resnet50


class Resnet50(nn.Module):
    def __init__(self):
        super().__init__()
        model=resnet50(pretrained=True)
        self.model=nn.Sequential(*list(model.children())[:-1])

    def forward(self,x):
        return self.model(x)

