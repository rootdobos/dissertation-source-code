import torch
from torch import nn
from torchvision.models.resnet import resnet50, resnext50_32x4d, ResNeXt50_32X4D_Weights


class Resnet50(nn.Module):
    def __init__(self):
        super().__init__()
        model=resnet50(pretrained=True)
        self.model=nn.Sequential(*list(model.children())[:-1])

    def forward(self,x):
        return self.model(x)
    

class Resnext50(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ResNeXt50_32X4D_Weights.IMAGENET1K_V2
        model=resnext50_32x4d(weights=weights)
        self.model=nn.Sequential(*list(model.children())[:-1])

    def forward(self,x):
        return self.model(x)

